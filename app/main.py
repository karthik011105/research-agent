import asyncio
import time
import uuid
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, Response, FileResponse
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
from app import progress
from app.monitoring import MAX_MONITORS, MIN_INTERVAL_HOURS, PostgresMonitorStore, schedule_due, what_changed
from app.learning import LearningService, PostgresLearningStore, outcome_parts
from app.security.attacks import ATTACKS, ATTACKS_BY_ID
from app.security.memory_poisoning import DEFENSE_SETTINGS
from app.security.rules import PostgresRuleStore, RuleBook
from app.security.service import execute_attack_run, execute_memory_run, propose_rule_for_attack
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


def memory_write_allowed(config, final_state: dict) -> bool:
    """MEMORY_VERIFIED_ONLY (Step 5c defense): only reports the critic fully approved are cached or remembered, so a
    report carrying an unverified (possibly injected) claim can't be served to later users."""
    if not config.memory_verified_only:
        return True
    verdicts = final_state.get("claim_verdicts") or []
    return bool(final_state.get("verified")) and all(v.get("decision") == "keep" for v in verdicts)


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
    monitor_id = data.get("monitor_id") or ""
    log = logging.getLogger(f"job.{job_id[:8]}")
    job_started = time.monotonic()
    progress_token = progress.use(progress.RedisProgress(redis_client, job_id))

    async def publish(result: dict) -> None:
        """Stores the job's result and ends its live progress stream with the matching final event."""
        await set_result(redis_client, config, job_id, result)
        status = result.get("status")
        await progress.emit(status if status in progress.FINAL_EVENTS else "done",
                            **({"origin": result.get("origin")} if status == "done" else {"error": result.get("error")}))

    try:
        attempt = await start_attempt(redis_client, job_id)
        if attempt > config.job_max_attempts:
            log.error(f"Giving up after {attempt - 1} interrupted attempts")
            await publish({
                "status": "error",
                "error": "The research job was interrupted repeatedly (server restarts). Please submit it again.",
            })
            return
        log.info(f"Starting job for topic: {topic}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        await progress.emit("started", topic=topic, attempt=attempt)

        # Greetings / "who are you?" are not research topics: answer instantly, spend no searches or LLM calls
        direct = quick_reply(topic)
        if direct:
            log.info("Not a research topic - instant reply")
            await publish(_direct_result(topic, direct))
            return

        # Fetch session history before any branch — agent always receives it
        session_history = await session_get(redis_client, session_id)

        # fresh=True skips both lookups so the report is always built from a live search
        origin = "pipeline"
        hit = None if fresh else await cache_get(redis_client, config, topic, session_id)
        if hit:
            log.info("Cache hit")
            origin = "cache"
            await progress.emit("memory_hit", where="cache")
        elif not fresh:
            hit = await ltm_search(config, topic, session_id)
            if hit:
                log.info("LTM hit")
                origin = "memory"
                await progress.emit("memory_hit", where="long-term memory")

        if hit:
            report_text, sources = hit["report"], hit.get("sources") or []
            live_sources, search_rounds, sub_questions, scraped_count = bool(sources), 0, [], 0
        else:
            log.info("Running multi-agent pipeline")
            # Find a related (not identical) previous report for the writer to reference
            ltm_context = await ltm_search_related(config, topic, session_id) or ""
            if ltm_context:
                log.info("Found related LTM context for writer agent")
            # session history makes the planner context-aware; ltm_context lets the writer build on prior research
            final_state = await graph.ainvoke(initial_state(topic, session_id, session_history, ltm_context))
            if final_state.get("direct_reply"):
                # Planner judged it not a research topic: no report to cache, remember or evaluate
                ok, reason, _ = await validate_output(config, final_state["direct_reply"])
                await publish(_direct_result(topic, final_state["direct_reply"]) if ok
                              else {"status": "blocked", "error": reason})
                return
            report_text = final_state["report"]
            live_sources = final_state["live_sources"]
            sources = final_state["sources"] if live_sources else []
            search_rounds = final_state["search_rounds"]
            sub_questions = [sq["question"] for sq in final_state["sub_questions"]] if live_sources else []
            scraped_count = final_state["scraped_count"]
            await progress.emit("safety_check")
            ok, reason, policies = await validate_output(config, report_text)
            if not ok:
                log.info(f"Output guardrail blocked the report ({policies}) - asking the writer for a safer draft")
                await progress.emit("defense", what="output_guardrail", policies=policies)
                report_text = await rewrite_for_safety(config, final_state, policies)
                ok, reason, policies = await validate_output(config, report_text)
            if not ok:
                await publish({"status": "blocked", "error": reason})
                return
            if memory_write_allowed(config, final_state):
                await cache_set(redis_client, config, topic, report_text, sources, session_id)
                await ltm_store(config, topic, report_text, str(uuid.uuid4()), sources, session_id)
            else:
                log.info("Report not stored in shared memory: the critic did not fully approve it")
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
            "monitor_id": monitor_id,
            # page text stays server-side; the client only needs what it shows
            "sources": [{k: s.get(k, "") for k in ("id", "title", "url", "published_date", "provider", "kind")} for s in sources],
        }

        if monitor_id:
            # Step 6b: compare with this monitor's previous report (code only, no LLM)
            try:
                result["changes"] = what_changed(await monitor_store.last_report(monitor_id), report_text)
                await monitor_store.record_run(monitor_id, job_id, report_text, result["changes"])
                await progress.emit("changes", new=len(result["changes"]["new"]), dropped=len(result["changes"]["dropped"]))
            except Exception as e:
                log.warning(f"Could not record monitor run: {e}")

        # Per-query evaluation runs automatically on every job
        asyncio.create_task(evaluate_report(config, job_id, topic, report_text, sources))

        if output_format == "pdf":
            pdf_bytes = generate_pdf(topic, report_text)
            result["pdf_base64"] = __import__("base64").b64encode(pdf_bytes).decode()
        elif output_format == "json":
            result["structured"] = generate_json_report(topic, report_text, job_id, datetime.utcnow(), result["sources"])

        await publish(result)
        log.info("Job completed successfully")
    except Exception as e:
        log.error(f"Job failed: {traceback.format_exc()}")
        await publish({"status": "error", "error": str(e)})
    finally:
        progress._emitter.reset(progress_token)
        if _inflight.pop(msg_id, None) is not None:   # not handed back during shutdown
            await ack_job(redis_client, config, msg_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, graph, bench_store, learning, rule_store, rulebook, monitor_store
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
    rule_store = PostgresRuleStore(get_pool())
    await rule_store.migrate()
    rulebook = RuleBook(rule_store)
    await rulebook.reload()
    graph = build_graph(config, redis_client, learning=learning, rulebook=rulebook)
    app.state.config = config
    monitor_store = PostgresMonitorStore(get_pool())
    await monitor_store.migrate()
    worker = asyncio.create_task(_worker_loop())
    scheduler = asyncio.create_task(_monitor_loop())
    yield
    worker.cancel()                        # stop taking new jobs first
    scheduler.cancel()
    await _hand_back_unfinished_jobs()
    await redis_client.aclose()
    await close_pool()


app = FastAPI(title="Research Agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
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


# ── Step 5: security research (run on demand; results in bench_runs) ──────────

rule_store: PostgresRuleStore | None = None
rulebook: RuleBook | None = None


def _background(coro):
    task = asyncio.create_task(coro)
    _bench_tasks.add(task)
    task.add_done_callback(_bench_tasks.discard)


class AttackRunRequest(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    topic_id: str = Field(min_length=1, max_length=60)
    topic: str = Field(min_length=3, max_length=500)
    attack_id: str
    critic_mode: str = "single"


@app.post("/security/attack-run", dependencies=[Depends(require_api_key)])
async def security_attack_run(req: AttackRunRequest):
    """5a/5b: one pipeline run with one poisoned page; records the stage where the attack stopped."""
    if req.attack_id not in ATTACKS_BY_ID:
        raise HTTPException(status_code=400, detail=f"attack_id must be one of {[a.id for a in ATTACKS]}")
    if req.critic_mode not in CRITIC_MODES:
        raise HTTPException(status_code=400, detail=f"critic_mode must be one of {CRITIC_MODES}")
    run = new_run(req.label, req.topic_id, req.topic, f"attack:{req.attack_id}:{req.critic_mode}")
    await bench_store.save_run(run)
    _background(execute_attack_run(config, redis_client, bench_store, run, req.attack_id, req.critic_mode, rulebook))
    return {"id": run["id"], "status": "running"}


class MemoryRunRequest(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    plant_topic: str = Field(min_length=3, max_length=300)
    victims: list[tuple[str, str]]          # (same | related | unrelated, topic)
    setting: str = "none"
    poison_verified: bool = False


@app.post("/security/memory-run", dependencies=[Depends(require_api_key)])
async def security_memory_run(req: MemoryRunRequest):
    """5c: plants a poisoned report, asks as another user, measures contamination, cleans up."""
    if req.setting not in DEFENSE_SETTINGS:
        raise HTTPException(status_code=400, detail=f"setting must be one of {list(DEFENSE_SETTINGS)}")
    run = new_run(req.label, "memory", req.plant_topic, f"memory:{req.setting}:{'verified' if req.poison_verified else 'unverified'}")
    await bench_store.save_run(run)
    _background(execute_memory_run(config, redis_client, bench_store, run, req.plant_topic, req.victims,
                                   req.setting, req.poison_verified))
    return {"id": run["id"], "status": "running"}


class RuleProposalRequest(BaseModel):
    attack_id: str


@app.post("/security/rules/propose", dependencies=[Depends(require_api_key)])
async def security_propose_rule(req: RuleProposalRequest):
    """5d: an LLM proposes a filter rule for an attack; code tests it. Nothing is applied until approved."""
    if req.attack_id not in ATTACKS_BY_ID:
        raise HTTPException(status_code=400, detail=f"attack_id must be one of {[a.id for a in ATTACKS]}")
    return await propose_rule_for_attack(config, bench_store, rule_store, req.attack_id)


@app.get("/security/rules", dependencies=[Depends(require_api_key)])
async def security_rules():
    return {"rules": await rule_store.list_rules(), "active_patterns": len(rulebook.patterns)}


@app.post("/security/rules/{rule_id}/{decision}", dependencies=[Depends(require_api_key)])
async def security_rule_decision(rule_id: str, decision: str):
    """The owner's approval step: only a rule that passed its tests can be approved."""
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be approve or reject")
    rule = await rule_store.get(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="no such rule")
    if decision == "approve" and not (rule.get("test") or {}).get("passed"):
        raise HTTPException(status_code=400, detail="this rule failed its tests and can't be approved")
    rule["status"] = "approved" if decision == "approve" else "rejected"
    await rule_store.save(rule)
    await rulebook.reload()
    return {"id": rule_id, "status": rule["status"], "active_patterns": len(rulebook.patterns)}


# ── Step 6a: live progress ───────────────────────────────────────────────────

@app.get("/stream/{job_id}", dependencies=[Depends(require_api_key)])
async def stream_progress(job_id: str):
    """Server-Sent Events: the job's progress as it happens (earlier events are replayed first)."""
    async def read(last_id: str, block_ms: int):
        return await progress.read_events(redis_client, job_id, last_id, block_ms)
    return StreamingResponse(progress.sse_events(read, job_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Step 6b: topic monitoring ────────────────────────────────────────────────

monitor_store: PostgresMonitorStore | None = None
MONITOR_CHECK_S = 600


async def _enqueue_monitor(monitor: dict) -> str:
    """A monitor run is a normal fresh research job, so it gets the live timeline, critics and memory like any other."""
    return await push_job(redis_client, config, monitor["topic"], monitor["session_id"], "text", fresh=True,
                          monitor_id=monitor["id"])


async def _monitor_loop():
    while True:
        try:
            queued = await schedule_due(monitor_store, redis_client, _enqueue_monitor)
            if queued:
                logger.info(f"Monitors queued: {queued}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Monitor check failed: {e}")
        await asyncio.sleep(MONITOR_CHECK_S)


class MonitorRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    interval_hours: int = Field(default=24, ge=MIN_INTERVAL_HOURS, le=24 * 30)
    session_id: str = ""


@app.post("/monitors", dependencies=[Depends(require_api_key)])
async def create_monitor(req: MonitorRequest):
    ok, reason = await validate_input(config, req.topic)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    if len(await monitor_store.list_monitors()) >= MAX_MONITORS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_MONITORS} monitored topics (each run uses Groq quota)")
    monitor = await monitor_store.create(req.topic, req.interval_hours, req.session_id or str(uuid.uuid4()))
    return {**monitor, "first_run": "within 10 minutes"}


@app.get("/monitors", dependencies=[Depends(require_api_key)])
async def list_monitors():
    monitors = await monitor_store.list_monitors()
    out = []
    for m in monitors:
        runs = await monitor_store.runs(m["id"], limit=1)
        out.append({**{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in m.items()},
                    "latest": runs[0] if runs else None})
    return {"monitors": out, "max": MAX_MONITORS}


@app.get("/monitors/{monitor_id}/runs", dependencies=[Depends(require_api_key)])
async def monitor_runs(monitor_id: str):
    return {"monitor_id": monitor_id, "runs": await monitor_store.runs(monitor_id)}


@app.delete("/monitors/{monitor_id}", dependencies=[Depends(require_api_key)])
async def delete_monitor(monitor_id: str):
    if not await monitor_store.deactivate(monitor_id):
        raise HTTPException(status_code=404, detail="no such monitor")
    return {"id": monitor_id, "active": False}
