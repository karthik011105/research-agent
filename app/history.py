"""Report history for the web app's History page: every finished research job, newest first.

Long-term memory (`reports`) only keeps reports the critic approved, because other users may be served them. The
history keeps every finished report, including ones answered from the cache or memory, so the owner can reopen
anything they asked. Greetings and other instant replies are not reports and are not kept.
"""
import json
from datetime import datetime

_LIST_COLUMNS = "job_id, topic, origin, live_sources, sources_count, notes_count, created_at, preview"


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
    }


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

    async def save(self, row: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO report_history (job_id, session_id, topic, report, sources, sub_questions, origin, "
                "live_sources, sources_count, notes_count, preview, created_at) "
                "VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8,$9,$10,$11,$12) ON CONFLICT (job_id) DO NOTHING",
                row["job_id"], row["session_id"], row["topic"], row["report"], json.dumps(row["sources"]),
                json.dumps(row["sub_questions"]), row["origin"], row["live_sources"], row["sources_count"],
                row["notes_count"], row["preview"], datetime.utcnow())

    async def list(self, limit: int = 50, query: str = "") -> list[dict]:
        async with self.pool.acquire() as conn:
            if query:
                rows = await conn.fetch(f"SELECT {_LIST_COLUMNS} FROM report_history WHERE topic ILIKE $1 "
                                        "ORDER BY created_at DESC LIMIT $2", f"%{query}%", limit)
            else:
                rows = await conn.fetch(f"SELECT {_LIST_COLUMNS} FROM report_history "
                                        "ORDER BY created_at DESC LIMIT $1", limit)
        return [{**dict(r), "created_at": r["created_at"].isoformat()} for r in rows]

    async def get(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM report_history WHERE job_id = $1", job_id)
        if not row:
            return None
        out = dict(row)
        for key in ("sources", "sub_questions"):
            if isinstance(out[key], str):
                out[key] = json.loads(out[key])
        out["created_at"] = out["created_at"].isoformat()
        return out
