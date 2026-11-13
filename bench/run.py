"""Runs the benchmark against the deployed app, one (topic, system) at a time. Safe to stop and re-run.

    python bench/run.py --url http://<alb_dns> [--label v1] [--set small | --topics n1,t4] [--systems multi,rag,single]

- Skips runs that already finished for this label; retries errors (up to --max-attempts) and runs left
  "running" by a deploy. Stops cleanly when Groq's daily limit is reached: run it again the next day.
- Per topic the systems run in the order multi -> rag -> single, so the search snapshots (the judge's evidence)
  are complete before the no-search system is judged.
- The API key comes from $API_KEY, else from the research-agent/config secret (never printed).
"""
import argparse
import json
import os
import pathlib
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="v1")
    ap.add_argument("--systems", default="multi,rag,single")
    ap.add_argument("--topics", default="", help="comma-separated topic ids (default: all)")
    ap.add_argument("--set", default="", help="a named topic set from topics.json, e.g. small")
    ap.add_argument("--max-runs", type=int, default=1000, help="stop after starting this many runs")
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

    started = 0
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
                if len(runs) >= args.max_attempts:
                    print(f"{t['id']}/{system}: gave up after {len(runs)} attempts: {runs[-1].get('error')}")
                    continue
                if started >= args.max_runs:
                    print(f"Reached --max-runs {args.max_runs}; run again to continue.")
                    return 0
                r = client.post(f"{base}/bench/run", json={"label": args.label, "topic_id": t["id"],
                                                           "topic": t["topic"], "system": system,
                                                           "trap": bool(t.get("trap"))})
                r.raise_for_status()
                run_id, started = r.json()["id"], started + 1
            result = wait(client, base, args.label, run_id)
            m, j = result.get("metrics") or {}, result.get("judge") or {}
            print(f"{t['id']}/{system}: {result['status']} {m.get('latency_s', '-')}s acc={j.get('accuracy')} "
                  f"calls={m.get('llm_calls')} searches={m.get('paid_searches')} {result.get('error') or ''}")
            if result["status"] == "error" and "daily limit" in (result.get("error") or "").lower():
                print("Groq daily limit reached - stopping. Run the same command again tomorrow to continue.")
                return 2
    print("Benchmark complete for this selection.")
    return 0


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
