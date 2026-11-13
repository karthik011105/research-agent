"""Runs the benchmark against the deployed app, one (topic, system) at a time. Safe to stop and re-run.

    python bench/run.py --url http://<alb_dns> [--label v1] [--set small | --topics n1,t4] [--systems multi,rag,single]

- Skips runs that already finished for this label; retries errors (up to --max-attempts) and runs left
  "running" by a deploy. Running out of Groq quota doesn't use up an attempt.
- When Groq's daily token limit is reached it stops at once: run the same command the next day. (The app already
  waits out per-minute limits; a "daily limit" error means the 24-hour token window is full. Groq's "retry in N
  min" hint is only for that one small request - a full research run needs ~45k tokens - and waiting on it just
  failed again, so the runner doesn't wait.)
- Per topic the systems run in the order multi -> rag -> single, so the search snapshots (the judge's evidence)
  are complete before the no-search system is judged.
- The API key comes from $API_KEY, else from the research-agent/config secret (never printed).
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
import httpx

TOPICS = pathlib.Path(__file__).with_name("topics.json")
STALE_AFTER_S = 25 * 60     # a "running" row older than this was lost (e.g. the task was replaced by a deploy)
RUN_TIMEOUT_S = 20 * 60


def api_key() -> str:
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    out = subprocess.run(["aws", "secretsmanager", "get-secret-value", "--secret-id", "research-agent/config",
                          "--region", "us-east-1", "--query", "SecretString", "--output", "text"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out).get("API_KEY", "")


def age_s(iso: str | None) -> float:
    if not iso:
        return 0
    started = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started).total_seconds()


def is_quota_error(run: dict) -> bool:
    return "daily limit" in (run.get("error") or "").lower()


def quota_retry_minutes(error: str | None) -> int | None:
    m = re.search(r"retry in (\d+) min", error or "")
    return int(m.group(1)) if m else None


def run_pass(client: httpx.Client, base: str, args, topics: list[dict], systems: list[str]) -> tuple[str, int]:
    """One pass over the selection. Returns ("done", 0) or ("quota_stop", 0)."""
    for t in topics:
        for system in systems:
            runs = [r for r in client.get(f"{base}/bench/runs", params={"label": args.label}).json()["runs"]
                    if r["topic_id"] == t["id"] and r["system"] == system]
            if any(r["status"] == "done" for r in runs):
                continue
            live = [r for r in runs if r["status"] == "running" and age_s(r["started_at"]) < STALE_AFTER_S]
            if live:
                print(f"{t['id']}/{system}: already running, waiting")
                run_id = live[0]["id"]
            else:
                # running out of quota says nothing about the run, so it doesn't count as an attempt
                attempts = [r for r in runs if r["status"] != "running" and not is_quota_error(r)]
                if len(attempts) >= args.max_attempts:
                    print(f"{t['id']}/{system}: gave up after {len(attempts)} attempts: {attempts[-1].get('error')}")
                    continue
                r = client.post(f"{base}/bench/run", json={"label": args.label, "topic_id": t["id"],
                                                           "topic": t["topic"], "system": system,
                                                           "trap": bool(t.get("trap"))})
                r.raise_for_status()
                run_id = r.json()["id"]
            result = wait(client, base, args.label, run_id)
            m, j = result.get("metrics") or {}, result.get("judge") or {}
            print(f"{t['id']}/{system}: {result['status']} {m.get('latency_s', '-')}s acc={j.get('accuracy')} "
                  f"calls={m.get('llm_calls')} searches={m.get('paid_searches')} {result.get('error') or ''}", flush=True)
            if result["status"] == "error" and is_quota_error(result):
                return "quota_stop", 0
    return "done", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="v1")
    ap.add_argument("--systems", default="multi,rag,single")
    ap.add_argument("--topics", default="", help="comma-separated topic ids (default: all)")
    ap.add_argument("--set", default="", help="a named topic set from topics.json, e.g. small")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base
    client = httpx.Client(headers={"X-API-Key": api_key()}, timeout=30)
    data = json.loads(TOPICS.read_text(encoding="utf-8"))
    topics = data["topics"]
    if args.set:
        topics = [t for t in topics if t["id"] in set(data["sets"][args.set])]
    if args.topics:
        wanted = set(args.topics.split(","))
        topics = [t for t in topics if t["id"] in wanted]
    systems = [s for s in ("multi", "rag", "single") if s in args.systems.split(",")]

    outcome, _ = run_pass(client, base, args, topics, systems)
    if outcome == "done":
        print("Benchmark complete for this selection.")
        return 0
    print("Groq daily token limit reached - stopping. Run the same command again tomorrow to continue.")
    return 2


def wait(client: httpx.Client, base: str, label: str, run_id: str) -> dict:
    deadline = time.time() + RUN_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(10)
        for r in client.get(f"{base}/bench/runs", params={"label": label}).json()["runs"]:
            if r["id"] == run_id and r["status"] != "running":
                return r
    return {"status": "timeout", "error": f"no result after {RUN_TIMEOUT_S // 60} min"}


if __name__ == "__main__":
    sys.exit(main())
