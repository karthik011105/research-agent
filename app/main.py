import asyncio
import uuid
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

from app.config import Config
from app.pool import init_pool, close_pool
from app.auth import require_api_key
from app.cache import cache_get, cache_set
from app.guardrails import validate_input, validate_output
from app.memory import session_add, session_get, ltm_search, ltm_search_related, ltm_store, ltm_diff, db_migrate
from app.queue import push_job, get_result, set_result, ensure_group, consume_jobs, ack_job
from app.agents import build_graph, initial_state
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


async def _worker_loop():
    await ensure_group(redis_client, config)
    while True:
        try:
            jobs = await consume_jobs(redis_client, config)
            for job in jobs:
                asyncio.create_task(_process_job(job["data"], job["msg_id"]))
        except Exception:
            await asyncio.sleep(1)


async def _process_job(data: dict, msg_id: str):
    job_id = data["job_id"]
    topic = data["topic"]
    session_id = data["session_id"]
    output_format = data.get("output_format", "text")
    fresh = data.get("fresh") == "1"
    log = logging.getLogger(f"job.{job_id[:8]}")
    try:
        log.info(f"Starting job for topic: {topic}")

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
            live_sources, search_rounds = bool(sources), 0
        else:
            log.info("Running multi-agent pipeline")
            # Find a related (not identical) previous report for the writer to reference
            ltm_context = await ltm_search_related(config, topic) or ""
            if ltm_context:
                log.info("Found related LTM context for writer agent")
            # session history makes the planner context-aware; ltm_context lets the writer build on prior research
            final_state = await graph.ainvoke(initial_state(topic, session_id, session_history, ltm_context))
            report_text = final_state["report"]
            live_sources = final_state["live_sources"]
            sources = final_state["sources"] if live_sources else []
            search_rounds = final_state["search_rounds"]
            ok, reason = await validate_output(config, report_text)
            if not ok:
                await set_result(redis_client, config, job_id, {"status": "blocked", "error": reason})
                return
            await cache_set(redis_client, config, topic, report_text, sources)
            await ltm_store(config, topic, report_text, str(uuid.uuid4()), sources)

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
            # page text stays server-side; the client only needs what it shows
            "sources": [{k: s.get(k, "") for k in ("id", "title", "url", "published_date", "provider")} for s in sources],
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
        await ack_job(redis_client, config, msg_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, graph
    redis_client = await aioredis.from_url(config.redis_url, decode_responses=True)
    await init_pool(config)
    await db_migrate(config)
    graph = build_graph(config, redis_client)
    app.state.config = config
    asyncio.create_task(_worker_loop())
    yield
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
    cache_keys = len([k async for k in redis_client.scan_iter("semantic:*")])
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
