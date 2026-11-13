"""Step 6a: live progress events for a research job.

Pipeline code calls `await emit("search_done", subq=1, results=8)`; outside a job it does nothing. The job runner
installs an emitter that appends events to a Redis stream per job (progress:<job_id>), so the browser can follow a
job whichever server task runs it, and a late subscriber still gets the earlier events. GET /stream/{job_id} relays
the stream as Server-Sent Events.

The ContextVar is copied into every task the job starts (LangGraph branches, asyncio.gather), so events from
parallel branches all reach the same job's stream.
"""
import json
import time
from contextvars import ContextVar

_emitter: ContextVar = ContextVar("progress_emitter", default=None)
STREAM_PREFIX = "progress:"
STREAM_TTL_S = 3600
STREAM_MAXLEN = 300
FINAL_EVENTS = ("done", "error", "blocked")


def use(emitter):
    """Install an emitter for the current job; returns a token for _emitter.reset()."""
    return _emitter.set(emitter)


async def emit(event_type: str, /, **data) -> None:
    """event_type is positional-only, so event data may use any field name (e.g. kind=...)."""
    emitter = _emitter.get()
    if emitter is None:
        return
    try:
        await emitter(event_type, data)
    except Exception:
        pass        # progress is best-effort: it must never break a research job


class RedisProgress:
    def __init__(self, redis, job_id: str):
        self.redis, self.key = redis, STREAM_PREFIX + job_id

    async def __call__(self, kind: str, data: dict) -> None:
        event = {"type": kind, "t": round(time.time(), 2), **data}
        await self.redis.xadd(self.key, {"e": json.dumps(event, default=str)}, maxlen=STREAM_MAXLEN, approximate=True)
        await self.redis.expire(self.key, STREAM_TTL_S)


async def read_events(redis, job_id: str, last_id: str, block_ms: int) -> list[tuple[str, dict]]:
    """Events after last_id, waiting up to block_ms for new ones."""
    reply = await redis.xread({STREAM_PREFIX + job_id: last_id}, block=block_ms, count=50)
    out = []
    for _, entries in reply or []:
        for entry_id, fields in entries:
            entry_id = entry_id.decode() if isinstance(entry_id, bytes) else entry_id
            raw = fields.get("e") or fields.get(b"e")
            out.append((entry_id, json.loads(raw.decode() if isinstance(raw, bytes) else raw)))
    return out


async def sse_events(read, job_id: str, keepalive_s: float = 15, max_s: float = 900):
    """Server-Sent Events text for a job, ending after a final event or max_s. `read(last_id, block_ms)` fetches.
    A comment line every keepalive_s keeps the connection open through the load balancer's 60 s idle timeout."""
    last, started = "0", time.monotonic()
    while time.monotonic() - started < max_s:
        events = await read(last, int(keepalive_s * 1000))
        if not events:
            yield ": keepalive\n\n"
            continue
        for entry_id, event in events:
            last = entry_id
            yield f"id: {entry_id}\nevent: progress\ndata: {json.dumps(event)}\n\n"
            if event.get("type") in FINAL_EVENTS:
                return
    yield 'event: progress\ndata: {"type": "timeout"}\n\n'
