"""Step 6b: topic monitoring - a saved topic is re-researched on a schedule, and each run shows what changed.

A scheduler loop inside the app (no extra AWS service) checks every few minutes for monitors that are due and
queues them as normal fresh research jobs. A Redis lock per monitor and time slot makes sure that when several app
tasks run, a monitor still runs once. When a monitored job finishes, its report is compared with the monitor's
previous report by code (no LLM tokens): claims that are new, and claims no longer reported.
"""
import json
import re
import uuid
from datetime import datetime, timedelta

MAX_MONITORS = 5
MIN_INTERVAL_HOURS = 6
_WORD = re.compile(r"[a-z0-9][a-z0-9.%\-]*")
_CITE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def _claims(report: str) -> list[str]:
    from app.agents import REVIEW_NOTES_HEADING
    from app.critics import split_claims
    # The fact-check notes quote flagged claims; they are not the report's own findings
    report = report.split(f"\n{REVIEW_NOTES_HEADING}")[0]
    cited, uncited = split_claims(report, max_claims=200)
    return [_CITE.sub("", c.text).strip() for c in cited] + uncited


def _words(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower()) if len(w) > 2 or w[0].isdigit()}


def _same(a: set[str], b: set[str], threshold: float) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= threshold


def what_changed(old_report: str | None, new_report: str, threshold: float = 0.5) -> dict:
    """Claims in the new report with no close match in the old one, and old claims that disappeared."""
    new_claims = _claims(new_report)
    if not old_report:
        return {"first_run": True, "new": [], "dropped": [], "unchanged": len(new_claims)}
    old_claims = _claims(old_report)
    old_words, new_words = [_words(c) for c in old_claims], [_words(c) for c in new_claims]
    added = [c for c, w in zip(new_claims, new_words) if not any(_same(w, o, threshold) for o in old_words)]
    dropped = [c for c, w in zip(old_claims, old_words) if not any(_same(w, n, threshold) for n in new_words)]
    return {"first_run": False, "new": added[:20], "dropped": dropped[:20],
            "unchanged": len(new_claims) - len(added)}


def is_due(monitor: dict, now: datetime) -> bool:
    if not monitor.get("active", True):
        return False
    last = monitor.get("last_run_at")
    return last is None or now - last >= timedelta(hours=monitor["interval_hours"])


def slot_key(monitor: dict, now: datetime) -> str:
    """Lock name for this monitor's current time slot: two app tasks checking the same slot get the same key."""
    slot = int(now.timestamp() // (monitor["interval_hours"] * 3600))
    return f"monitor_lock:{monitor['id']}:{slot}"


async def schedule_due(store, redis, enqueue, now: datetime | None = None) -> list[str]:
    """Queues every due monitor once (Redis SET NX per slot). Returns the queued monitor ids."""
    now = now or datetime.utcnow()
    queued = []
    for m in await store.list_monitors():
        if not is_due(m, now):
            continue
        if not await redis.set(slot_key(m, now), "1", nx=True, ex=m["interval_hours"] * 3600):
            continue                                    # another task already queued this slot
        job_id = await enqueue(m)
        await store.mark_started(m["id"], now, job_id)
        queued.append(m["id"])
    return queued


class PostgresMonitorStore:
    def __init__(self, pool):
        self.pool = pool

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS monitors (
                    id TEXT PRIMARY KEY, topic TEXT NOT NULL, interval_hours INT NOT NULL, session_id TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    last_run_at TIMESTAMP, last_job_id TEXT)""")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS monitor_runs (
                    id TEXT PRIMARY KEY, monitor_id TEXT NOT NULL, job_id TEXT NOT NULL, report TEXT NOT NULL,
                    changes JSONB NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT NOW())""")
            await conn.execute("CREATE INDEX IF NOT EXISTS monitor_runs_monitor_idx ON monitor_runs (monitor_id, created_at)")

    async def create(self, topic: str, interval_hours: int, session_id: str) -> dict:
        monitor = {"id": uuid.uuid4().hex[:12], "topic": topic, "interval_hours": interval_hours,
                   "session_id": session_id, "active": True}
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO monitors (id, topic, interval_hours, session_id) VALUES ($1,$2,$3,$4)",
                               monitor["id"], topic, interval_hours, session_id)
        return monitor

    async def list_monitors(self, active_only: bool = True) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM monitors" + (" WHERE active" if active_only else "") +
                                    " ORDER BY created_at")
        return [dict(r) for r in rows]

    async def deactivate(self, monitor_id: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("UPDATE monitors SET active = FALSE WHERE id = $1", monitor_id)
        return result.endswith("1")

    async def mark_started(self, monitor_id: str, when: datetime, job_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE monitors SET last_run_at = $2, last_job_id = $3 WHERE id = $1",
                               monitor_id, when, job_id)

    async def last_report(self, monitor_id: str) -> str | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT report FROM monitor_runs WHERE monitor_id = $1 "
                                      "ORDER BY created_at DESC LIMIT 1", monitor_id)
        return row["report"] if row else None

    async def record_run(self, monitor_id: str, job_id: str, report: str, changes: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO monitor_runs (id, monitor_id, job_id, report, changes) "
                               "VALUES ($1,$2,$3,$4,$5::jsonb)", uuid.uuid4().hex, monitor_id, job_id, report,
                               json.dumps(changes))

    async def runs(self, monitor_id: str, limit: int = 20) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT job_id, changes, created_at FROM monitor_runs WHERE monitor_id = $1 "
                                    "ORDER BY created_at DESC LIMIT $2", monitor_id, limit)
        return [{"job_id": r["job_id"], "changes": json.loads(r["changes"]), "at": r["created_at"].isoformat()}
                for r in rows]
