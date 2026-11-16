"""Report history for the web app's History page: every finished research job, newest first.

Long-term memory (`reports`) only keeps reports the critic approved, because other users may be served them. The
history keeps every finished report, including ones answered from the cache or memory, so the owner can reopen
anything they asked. Greetings and other instant replies are not reports and are not kept.
"""
import json
from datetime import datetime

_LIST_COLUMNS = ("job_id, topic, origin, live_sources, sources_count, notes_count, created_at, preview, trust, "
                 "review_status, review_note, reviewed_at")
REVIEW_DECISIONS = {"approve": "approved", "reject": "rejected"}


def history_record(job_id: str, session_id: str, result: dict) -> dict | None:
    """The row to keep for a finished job, or None if there is nothing worth listing."""
    if result.get("status") != "done" or result.get("origin") == "assistant" or not result.get("report"):
        return None
    report = result["report"]
    body = report.split("\n## Sources")[0]
    preview = " ".join(line.strip("#*-• ").strip() for line in body.splitlines()
                       if line.strip() and not line.lstrip().startswith("#") and "summary" not in line.lower()[:30])
    return {
        "job_id": job_id,
        "session_id": session_id,
        "topic": result.get("topic", ""),
        "report": report,
        "sources": result.get("sources") or [],
        "sub_questions": result.get("sub_questions") or [],
        "origin": result.get("origin", "pipeline"),
        "live_sources": bool(result.get("live_sources")),
        "sources_count": int(result.get("sources_count") or 0),
        "notes_count": len(result.get("review_notes") or []),
        "preview": preview[:280],
        "trust": result.get("trust"),
        "review_status": (result.get("review") or {}).get("status", "not_needed"),
    }


def _json(value):
    """asyncpg returns JSONB as text unless a codec is set."""
    return json.loads(value) if isinstance(value, str) else value


def _row(row) -> dict:
    out = dict(row)
    for key in ("sources", "sub_questions", "trust"):
        if key in out:
            out[key] = _json(out[key])
    for key in ("created_at", "reviewed_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    out["review_status"] = out.get("review_status") or "not_needed"     # reports saved before reviews existed
    return out


async def review_report(store, job_id: str, decision: str, note: str, remember) -> dict | None:
    """A reviewer approves or rejects a report held for review. Only an approved report is put into shared memory
    (`remember(row)`): a doubtful report never reaches another user unless a person said it may."""
    row = await store.decide(job_id, REVIEW_DECISIONS[decision], note.strip()[:500])
    if row is not None and row["review_status"] == "approved":
        await remember(row)
    return row


class PostgresHistoryStore:
    def __init__(self, pool):
        self.pool = pool

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS report_history (
                    job_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, topic TEXT NOT NULL, report TEXT NOT NULL,
                    sources JSONB NOT NULL, sub_questions JSONB NOT NULL, origin TEXT NOT NULL,
                    live_sources BOOLEAN NOT NULL, sources_count INT NOT NULL, notes_count INT NOT NULL,
                    preview TEXT NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT NOW())""")
            await conn.execute("CREATE INDEX IF NOT EXISTS report_history_created_idx ON report_history (created_at DESC)")
            await conn.execute("ALTER TABLE report_history ADD COLUMN IF NOT EXISTS trust JSONB")
            # human-in-the-loop review: not_needed | pending | approved | rejected
            await conn.execute("ALTER TABLE report_history ADD COLUMN IF NOT EXISTS review_status TEXT")
            await conn.execute("ALTER TABLE report_history ADD COLUMN IF NOT EXISTS review_note TEXT")
            await conn.execute("ALTER TABLE report_history ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP")

    async def save(self, row: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO report_history (job_id, session_id, topic, report, sources, sub_questions, origin, "
                "live_sources, sources_count, notes_count, preview, created_at, trust, review_status) "
                "VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8,$9,$10,$11,$12,$13::jsonb,$14) "
                "ON CONFLICT (job_id) DO NOTHING",
                row["job_id"], row["session_id"], row["topic"], row["report"], json.dumps(row["sources"]),
                json.dumps(row["sub_questions"]), row["origin"], row["live_sources"], row["sources_count"],
                row["notes_count"], row["preview"], datetime.utcnow(),
                json.dumps(row["trust"]) if row.get("trust") is not None else None,
                row.get("review_status", "not_needed"))

    async def list(self, limit: int = 50, query: str = "", review: str = "") -> list[dict]:
        where, args = [], []
        if query:
            args.append(f"%{query}%")
            where.append(f"topic ILIKE ${len(args)}")
        if review == "decided":
            where.append("review_status IN ('approved', 'rejected')")
        elif review:
            args.append(review)
            where.append(f"review_status = ${len(args)}")
        args.append(limit)
        sql = (f"SELECT {_LIST_COLUMNS} FROM report_history {'WHERE ' + ' AND '.join(where) if where else ''} "
               f"ORDER BY created_at DESC LIMIT ${len(args)}")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row(r) for r in rows]

    async def decide(self, job_id: str, status: str, note: str) -> dict | None:
        """Records a reviewer's decision on a report that is waiting for one; None if there is no such report."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE report_history SET review_status=$2, review_note=$3, reviewed_at=$4 "
                "WHERE job_id=$1 AND review_status='pending' RETURNING *", job_id, status, note, datetime.utcnow())
        return _row(row) if row else None

    async def get(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM report_history WHERE job_id = $1", job_id)
        if not row:
            return None
        return _row(row)
