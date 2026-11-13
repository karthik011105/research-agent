import asyncio
import time
import uuid
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

from app.config import Config
from app.pool import init_pool, close_pool, get_pool
from app.bench import CRITIC_MODES, SYSTEMS, PostgresBenchStore, execute_critic_eval, execute_run, new_run
from app.learning import LearningService, PostgresLearningStore, outcome_parts
from app.auth import require_api_key
from app.cache import cache_get, cache_set
from app.guardrails import validate_input, validate_output
from app.memory import session_add, session_get, ltm_search, ltm_search_related, ltm_store, ltm_diff, db_migrate
from app.queue import (
    push_job, get_result, set_result, ensure_group, consume_jobs, ack_job, requeue_job, start_attempt,
)
from app.agents import build_graph, initial_state, quick_reply, rewrite_for_safety
from app.output import generate_pdf, generate_json_report, get_report_diff
from app.eval import evaluate_report, run_batch_evaluation, fetch_recent_topics

config = Config()
redis_client: aioredis.Redis = None
graph = None


async def _rate_limit(request: Request) -> None:
    client_ip = request.client.host
    key = f"ratelimit:{client_ip}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, config.rate_limit_window)
    if count > config.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


# Jobs this worker has taken from the queue and not finished: msg_id -> job data
_inflight: dict[str, dict] = {}


async def _worker_loop():
    await ensure_group(redis_client, config)
    while True:
        try:
            jobs = await consume_jobs(redis_client, config)
            for job in jobs:
                _inflight[job["msg_id"]] = job["data"]
                asyncio.create_task(_process_job(job["data"], job["msg_id"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1)


async def _hand_back_unfinished_jobs():
    """On shutdown (deploy, scale-in, pause), put running jobs back on the queue for the next worker."""
    for msg_id, data in list(_inflight.items()):
        try:
            await requeue_job(redis_client, config, msg_id, data)
            _inflight.pop(msg_id, None)
            logger.info(f"Handed job {data.get('job_id', '?')[:8]} back to the queue")
        except Exception as e:
            logger.error(f"Could not hand back job {data.get('job_id', '?')[:8]}: {e}")


def _direct_result(topic: str, reply: str) -> dict:
    """Result shape for inputs answered without research (greetings, questions about the assistant)."""
    return {
        "status": "done", "topic": topic, "report": reply, "diff": None, "origin": "assistant",
        "live_sources": False, "search_rounds": 0, "sources_count": 0, "sub_questions": [],
        "providers_used": [], "scraped_count": 0, "sources": [],
    }


async def _process_job(data: dict, msg_id: str):
    job_id = data["job_id"]
    topic = data["topic"]
    session_id = data["session_id"]
    output_format = data.get("output_format", "text")
    fresh = data.get("fresh") == "1"
    log = logging.getLogger(f"job.{job_id[:8]}")
    job_started = time.monotonic()
    try:
        attempt = await start_attempt(redis_client, job_id)
        if attempt > config.job_max_attempts:
            log.error(f"Giving up after {attempt - 1} interrupted attempts")
            await set_result(redis_client, config, job_id, {
                "status": "error",
                "error": "The research job was interrupted repeatedly (server restarts). Please submit it again.",
            })
            return
        log.info(f"Starting job for topic: {topic}" + (f" (attempt {attempt})" if attempt > 1 else ""))

        # Greetings / "who are you?" are not research topics: answer instantly, spend no searches or LLM calls
        direct = quick_reply(topic)
        if direct:
            log.info("Not a research topic - instant reply")
            await set_result(redis_client, config, job_id, _direct_result(topic, direct))
            return

        # Fetch session history before any branch — agent always receives it
        session_history = await session_get(redis_client, session_id)

        # fresh=True skips both lookups so the report is always built from a live search
        origin = "pipeline"
        hit = None if fresh else await cache_get(redis_client, config, topic)
        if hit:
            log.info("Cache hit")
            origin = "cache"
        elif not fresh:
            hit = await ltm_search(config, topic)
            if hit:
                log.info("LTM hit")
                origin = "memory"

        if hit:
            report_text, sources = hit["report"], hit.get("sources") or []
            live_sources, search_rounds, sub_questions, scraped_count = bool(sources), 0, [], 0
        else:
            log.info("Running multi-agent pipeline")
            # Find a related (not identical) previous report for the writer to reference
            ltm_context = await ltm_search_related(config, topic) or ""
            if ltm_context:
                log.info("Found related LTM context for writer agent")
            # session history makes the planner context-aware; ltm_context lets the writer build on prior research
            final_state = await graph.ainvoke(initial_state(topic, session_id, session_history, ltm_context))
            if final_state.get("direct_reply"):
                # Planner judged it not a research topic: no report to cache, remember or evaluate
                ok, reason, _ = await validate_output(config, final_state["direct_reply"])
                await set_result(redis_client, config, job_id,
                                 _direct_result(topic, final_state["direct_reply"]) if ok
                                 else {"status": "blocked", "error": reason})
                return
            report_text = final_state["report"]
            live_sources = final_state["live_sources"]
            sources = final_state["sources"] if live_sources else []
            search_rounds = final_state["search_rounds"]
            sub_questions = [sq["question"] for sq in final_state["sub_questions"]] if live_sources else []
            scraped_count = final_state["scraped_count"]
            ok, reason, policies = await validate_output(config, report_text)
            if not ok:
                log.info(f"Output guardrail blocked the report ({policies}) - asking the writer for a safer draft")
                report_text = await rewrite_for_safety(config, final_state, policies)
                ok, reason, policies = await validate_output(config, report_text)
            if not ok:
                await set_result(redis_client, config, job_id, {"status": "blocked", "error": reason})
                return
            await cache_set(redis_client, config, topic, report_text, sources)
            await ltm_store(config, topic, report_text, str(uuid.uuid4()), sources)
            writer_variant = final_state.get("writer_variant") or ""
            if writer_variant and learning:
                # Step 4c: the bandit learns from this report's outcome (👍/👎 can add to it later)
                try:
                    parts = outcome_parts(final_state, report_text, sources, time.monotonic() - job_started)
                    reward = await learning.record_report(job_id, writer_variant, parts)
                    log.info(f"Writer variant {writer_variant}: reward {reward} from {parts}")
                except Exception as e:
                    log.warning(f"Could not record writer reward: {e}")

        await session_add(redis_client, config, session_id, "assistant", report_text[:config.session_content_truncate])
        diff = await ltm_diff(config, topic)
        result: dict = {
            "status": "done",
            "topic": topic,
            "report": report_text,
            "diff": diff,
            "origin": origin,
            "live_sources": live_sources,
            "search_rounds": search_rounds,
            "sources_count": len(sources),
            "sub_questions": sub_questions,
            "providers_used": sorted({s.get("provider", "") for s in sources} - {""}),
            "scraped_count": scraped_count,
            "writer_variant": (final_state.get("writer_variant") or "") if origin == "pipeline" else "",
            # page text stays server-side; the client only needs what it shows
            "sources": [{k: s.get(k, "") for k in ("id", "title", "url", "published_date", "provider", "kind")} for s in sources],
        }

        # Per-query evaluation runs automatically on every job
        asyncio.create_task(evaluate_report(config, job_id, topic, report_text, sources))

        if output_format == "pdf":
            pdf_bytes = generate_pdf(topic, report_text)
            result["pdf_base64"] = __import__("base64").b64encode(pdf_bytes).decode()
        elif output_format == "json":
            result["structured"] = generate_json_report(topic, report_text, job_id, datetime.utcnow(), result["sources"])

        await set_result(redis_client, config, job_id, result)
        log.info("Job completed successfully")
    except Exception as e:
        log.error(f"Job failed: {traceback.format_exc()}")
        await set_result(redis_client, config, job_id, {"status": "error", "error": str(e)})
    finally:
        if _inflight.pop(msg_id, None) is not None:   # not handed back during shutdown
            await ack_job(redis_client, config, msg_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, graph, bench_store, learning
    if not config.guardrails_enabled:
        logger.warning("Bedrock guardrails are DISABLED (GUARDRAILS_ENABLED=false) - local development only")
    redis_client = await aioredis.from_url(config.redis_url, decode_responses=True)
    await init_pool(config)
    await db_migrate(config)
    bench_store = PostgresBenchStore(get_pool())
    await bench_store.migrate()
    learning_store = PostgresLearningStore(get_pool())
    await learning_store.migrate()
    learning = LearningService(learning_store, config.learning_enabled, config.learning_explore)
    await learning.load()
    graph = build_graph(config, redis_client, learning=learning)
    app.state.config = config
    worker = asyncio.create_task(_worker_loop())
    yield
    worker.cancel()                        # stop taking new jobs first
    await _hand_back_unfinished_jobs()
    await redis_client.aclose()
    await close_pool()


app = FastAPI(title="Research Agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str
    session_id: str = ""
    output_format: str = "text"
    fresh: bool = False  # skip cache and long-term memory, always search the live web


@app.get("/")
async def frontend():
    return FileResponse("/app/index.html")


@app.get("/health")
async def health():
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "error",
    }


@app.post("/research", dependencies=[Depends(require_api_key), Depends(_rate_limit)])
async def start_research(req: ResearchRequest):
    ok, reason = await validate_input(config, req.topic)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    session_id = req.session_id or str(uuid.uuid4())
    await session_add(redis_client, config, session_id, "user", req.topic)
    job_id = await push_job(redis_client, config, req.topic, session_id, req.output_format, req.fresh)
    return {"job_id": job_id, "session_id": session_id}


@app.get("/result/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job_result(job_id: str):
    result = await get_result(redis_client, config, job_id)
    if result is None:
        return {"status": "pending"}
    return result


@app.get("/session/{session_id}", dependencies=[Depends(require_api_key)])
async def get_session(session_id: str):
    messages = await session_get(redis_client, session_id)
    return {"session_id": session_id, "messages": messages}


@app.get("/diff/{topic}", dependencies=[Depends(require_api_key)])
async def report_diff(topic: str):
    diff = await get_report_diff(config, topic)
    return {"topic": topic, "diff": diff or "No previous report found."}


@app.get("/result/{job_id}/pdf", dependencies=[Depends(require_api_key)])
async def download_pdf(job_id: str):
    result = await get_result(redis_client, config, job_id)
    if not result or result.get("status") != "done":
        raise HTTPException(status_code=404, detail="Report not ready")
    pdf_bytes = generate_pdf(result.get("topic", "Report"), result["report"])
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={job_id}.pdf"},
    )


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def stats():
    info = await redis_client.info()
    keys = await redis_client.dbsize()
    cache_keys = await redis_client.hlen("semantic:index")
    session_keys = len([k async for k in redis_client.scan_iter("session:*")])
    search_cache_keys = len([k async for k in redis_client.scan_iter("search:*")])
    searches_today = int(await redis_client.get(f"search_budget:{date.today().isoformat()}") or 0)
    return {
        "search": {
            "providers": [name for name, key in (("tavily", config.tavily_api_key), ("brave", config.brave_api_key)) if key],
            "searches_today": searches_today,
            "daily_limit": config.search_daily_limit,
            "cached_queries": search_cache_keys,
        },
        "redis": {
            "total_keys": keys,
            "cache_entries": cache_keys,
            "active_sessions": session_keys,
            "memory_used_mb": round(info["used_memory"] / 1024 / 1024, 2),
            "connected_clients": info["connected_clients"],
            "uptime_hours": round(info["uptime_in_seconds"] / 3600, 1),
        },
        "tensorzero_url": config.tensorzero_url,
        "guardrail_id": config.bedrock_guardrail_id,
    }


@app.get("/evaluate/{job_id}", dependencies=[Depends(require_api_key)])
async def evaluate_job(job_id: str):
    result = await get_result(redis_client, config, job_id)
    if not result or result.get("status") != "done":
        raise HTTPException(status_code=404, detail="Job not done yet")
    scores = await evaluate_report(config, job_id, result["topic"], result["report"], result.get("sources", []))
    return {"job_id": job_id, "topic": result["topic"], "scores": scores}


class BatchEvalRequest(BaseModel):
    topics: list[str] = []


@app.post("/run-evaluation", dependencies=[Depends(require_api_key)])
async def trigger_batch_evaluation(req: BatchEvalRequest):
    topics = req.topics if req.topics else await fetch_recent_topics()
    if not topics:
        raise HTTPException(status_code=400, detail="No topics found. Submit at least one research job first.")
    asyncio.create_task(run_batch_evaluation(config, graph, topics))
    return {"message": "Batch evaluation started in background", "topics": len(topics)}


# ── Benchmark (Step 2): driven by bench/run.py, results read by bench/report.py ──

bench_store: PostgresBenchStore | None = None
_bench_tasks: set[asyncio.Task] = set()


class BenchRunRequest(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    topic_id: str = Field(min_length=1, max_length=60)
    topic: str = Field(min_length=3, max_length=500)
    system: str
    trap: bool = False


@app.post("/bench/run", dependencies=[Depends(require_api_key)])
async def bench_run(req: BenchRunRequest):
    """Starts one (topic, system) benchmark run in the background - runs take minutes, longer than the ALB timeout."""
    if req.system not in SYSTEMS:
        raise HTTPException(status_code=400, detail=f"system must be one of {SYSTEMS}")
    run = new_run(req.label, req.topic_id, req.topic, req.system)
    await bench_store.save_run(run)
    task = asyncio.create_task(execute_run(config, redis_client, bench_store, run, req.trap))
    _bench_tasks.add(task)                 # keep a reference: unreferenced tasks can be garbage-collected
    task.add_done_callback(_bench_tasks.discard)
    return {"id": run["id"], "status": "running"}


class CriticEvalRequest(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    topic_id: str = Field(min_length=1, max_length=60)
    mode: str
    learn: bool = False      # Step 4a/4b: train critic trust and the debate policy from the labels


@app.post("/bench/critic-eval", dependencies=[Depends(require_api_key)])
async def bench_critic_eval(req: CriticEvalRequest):
    """Step 3: scores one critic setup on error-injected claims built from a topic's saved sources."""
    if req.mode not in CRITIC_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {CRITIC_MODES}")
    run = new_run(req.label, req.topic_id, f"critic eval ({req.mode})", f"critic_{req.mode}")
    await bench_store.save_run(run)
    task = asyncio.create_task(execute_critic_eval(config, bench_store, run, req.mode,
                                                   learning if req.learn else None))
    _bench_tasks.add(task)
    task.add_done_callback(_bench_tasks.discard)
    return {"id": run["id"], "status": "running"}


@app.get("/bench/runs", dependencies=[Depends(require_api_key)])
async def bench_runs(label: str, full: bool = False):
    return {"label": label, "runs": await bench_store.list_runs(label, full)}


# ── Step 4: learning from scores ─────────────────────────────────────────────

learning: LearningService | None = None


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1)


@app.post("/feedback/{job_id}", dependencies=[Depends(require_api_key)])
async def feedback(job_id: str, req: FeedbackRequest):
    """👍 (+1) / 👎 (-1) on a report: one more reward for the writer variant that wrote it."""
    if req.rating == 0:
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")
    out = await learning.apply_feedback(job_id, req.rating)
    if out is None:
        return {"recorded": False, "detail": "No learning record for this job (cached report, or learning off)."}
    return {"recorded": True, **out}


@app.get("/stats/learning", dependencies=[Depends(require_api_key)])
async def learning_stats():
    return await learning.stats()
