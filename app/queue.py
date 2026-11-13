import json
import uuid
import redis.asyncio as aioredis
from app.config import Config


async def push_job(redis: aioredis.Redis, config: Config, topic: str, session_id: str, output_format: str, fresh: bool = False) -> str:
    job_id = str(uuid.uuid4())
    await redis.xadd(config.stream_key, {
        "job_id": job_id,
        "topic": topic,
        "session_id": session_id,
        "output_format": output_format,
        "fresh": "1" if fresh else "0",
    })
    return job_id


async def get_result(redis: aioredis.Redis, config: Config, job_id: str) -> dict | None:
    data = await redis.get(f"result:{job_id}")
    return json.loads(data) if data else None


async def set_result(redis: aioredis.Redis, config: Config, job_id: str, result: dict) -> None:
    await redis.setex(f"result:{job_id}", config.result_ttl, json.dumps(result))


async def ensure_group(redis: aioredis.Redis, config: Config) -> None:
    try:
        await redis.xgroup_create(config.stream_key, config.consumer_group, id="0", mkstream=True)
    except Exception:
        pass


async def claim_stale_jobs(redis: aioredis.Redis, config: Config) -> list[dict]:
    """Jobs delivered to a worker that died without finishing (task crash, OOM, SIGKILL) stay pending in the
    consumer group forever. Take over one that nobody has touched for job_reclaim_idle_ms."""
    _, claimed, _ = await redis.xautoclaim(
        config.stream_key, config.consumer_group, config.consumer_name,
        min_idle_time=config.job_reclaim_idle_ms, start_id="0-0", count=1,
    )
    return [{"msg_id": msg_id, "data": data} for msg_id, data in claimed if data]


async def requeue_job(redis: aioredis.Redis, config: Config, msg_id: str, data: dict) -> None:
    """Hands an unfinished job back to the queue (as a new message) on shutdown, so another worker starts it
    right away instead of after the reclaim timeout. Add + ack in one transaction: never lost, never doubled."""
    async with redis.pipeline(transaction=True) as pipe:
        pipe.xadd(config.stream_key, data)
        pipe.xack(config.stream_key, config.consumer_group, msg_id)
        await pipe.execute()


async def start_attempt(redis: aioredis.Redis, job_id: str) -> int:
    """Counts how many times a job has been started, across workers and restarts."""
    key = f"job_attempts:{job_id}"
    attempts = await redis.incr(key)
    if attempts == 1:
        await redis.expire(key, 86400)
    return attempts


async def consume_jobs(redis: aioredis.Redis, config: Config) -> list[dict]:
    stale = await claim_stale_jobs(redis, config)
    if stale:
        return stale
    messages = await redis.xreadgroup(
        config.consumer_group,
        config.consumer_name,
        {config.stream_key: ">"},
        count=1,
        block=5000,
    )
    if not messages:
        return []
    jobs = []
    for _, entries in messages:
        for msg_id, data in entries:
            jobs.append({"msg_id": msg_id, "data": data})
    return jobs


async def ack_job(redis: aioredis.Redis, config: Config, msg_id: str) -> None:
    await redis.xack(config.stream_key, config.consumer_group, msg_id)
