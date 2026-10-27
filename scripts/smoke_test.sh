#!/bin/bash
# End-to-end check of a deployed stack: runs one fresh (live web) research job and checks the result.
# Usage: ./scripts/smoke_test.sh <alb_dns> [api_key] [topic]
#   alb_dns comes from: cd terraform && terraform output -raw alb_dns
set -e

HOST=${1:?usage: smoke_test.sh <alb_dns> [api_key] [topic]}
API_KEY=${2:-}
TOPIC=${3:-"Latest developments in AI chips this month"}
BASE="http://$HOST"
AUTH=()
[ -n "$API_KEY" ] && AUTH=(-H "X-API-Key: $API_KEY")

echo "1/3 Health check..."
curl -sf "$BASE/health"
echo

echo "2/3 Submitting fresh research job: $TOPIC"
BODY=$(python -c 'import json,sys; print(json.dumps({"topic": sys.argv[1], "fresh": True}))' "$TOPIC")
JOB=$(curl -sf -X POST "$BASE/research" -H "Content-Type: application/json" "${AUTH[@]}" -d "$BODY")
JOB_ID=$(echo "$JOB" | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "   job_id=$JOB_ID"

echo "3/3 Waiting for result (up to 5 min)..."
for _ in $(seq 1 60); do
  RESULT=$(curl -sf "$BASE/result/$JOB_ID" "${AUTH[@]}")
  STATUS=$(echo "$RESULT" | python -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$STATUS" != "pending" ] && break
  sleep 5
done

python - "$RESULT" <<'PY'
import json, re, sys
r = json.loads(sys.argv[1])
status, sources = r.get("status"), r.get("sources", [])
problems = []
if status != "done":
    problems.append(f"status={status} error={r.get('error')}")
else:
    if not r.get("live_sources"):
        problems.append("live_sources is false - is TAVILY_API_KEY set in Secrets Manager?")
    if not sources:
        problems.append("no sources returned")
    if not re.search(r"\[\d+", r.get("report", "")):
        problems.append("report has no [n] citations")
print(f"   status={status} sources={r.get('sources_count')} search_rounds={r.get('search_rounds')} origin={r.get('origin')}")
for s in sources[:5]:
    print(f"   [{s['id']}] {s['title'][:70]} - {s['url']}")
if problems:
    print("FAIL:")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("PASS")
PY
