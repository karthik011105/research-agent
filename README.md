# Autonomous Research Agent

Give it a topic → it researches, writes a full report, safety-checks it, caches it, and remembers it. Built on AWS with a real multi-agent pipeline, red teaming, and LLM evaluation on every request.

---

## What It Uses

| Component | What It Does |
|---|---|
| **FastAPI** | REST API — receives topics, returns reports |
| **LangGraph** | Multi-agent pipeline: Plan → Web Search → Extract → Gap Check → Write → Verify |
| **Tavily** (+ optional Brave) | Live web search — reports are built from today's pages, with `[n]` citations |
| **TensorZero** | LLM gateway — routes to GPT-4o, falls back to Groq Llama-3 |
| **AWS Bedrock Guardrails** | Blocks harmful input and output automatically |
| **Redis (ElastiCache)** | Semantic cache + session memory + job queue |
| **PostgreSQL + pgvector (RDS)** | Long-term memory — stores reports as vectors, enables semantic search |
| **LangSmith** | Traces every agent run + LLM-as-judge scores every report |
| **PyRIT 0.14.0** | Automated red team attacks — jailbreak, XPIA, crescendo, skeleton key |
| **Terraform** | Creates all AWS infrastructure with one command |
| **GitHub Actions** | Builds Docker images and deploys to ECS automatically on every push |

---

## File Structure

```
PROJECT/
├── app/
│   ├── main.py           API, background worker, all endpoints
│   ├── agents.py         LangGraph multi-agent graph
│   ├── tools/search.py   Live web search (Tavily, Brave fallback), search cache, daily budget
│   ├── cache.py          Redis semantic cache
│   ├── guardrails.py     Bedrock safety checks
│   ├── memory.py         Session memory (Redis) + long-term memory (pgvector)
│   ├── queue.py          Redis Streams job queue
│   ├── output.py         PDF export, JSON report, report diff
│   ├── eval.py           LangSmith LLM-as-judge evaluation
│   ├── config.py         Loads everything from AWS Secrets Manager
│   ├── auth.py           API key middleware
│   ├── retry.py          Exponential backoff for LLM calls
│   ├── pool.py           PostgreSQL connection pool
│   └── Dockerfile
├── pyrit_dashboard/
│   ├── main.py           Red team attack dashboard (PyRIT 0.14.0)
│   ├── requirements.txt
│   └── Dockerfile
├── tensorzero/
│   ├── tensorzero.toml   LLM routing config with system prompts
│   └── Dockerfile
├── terraform/
│   └── main.tf           All AWS infrastructure
├── .github/workflows/
│   └── deploy.yml        CI/CD pipeline with rollback on failure
├── bootstrap.bat         One-time backend setup (Windows)
├── bootstrap.sh          One-time backend setup (Mac/Linux)
├── requirements.txt      Python dependencies
├── requirements-dev.txt  Test dependencies (no torch)
├── tests/                pytest suite (runs in CI before every deploy)
├── scripts/smoke_test.sh End-to-end check of a deployed stack
├── index.html            Frontend UI
└── README.md
```

---



### How live search works

1. **Plan** — the LLM turns the topic into up to 4 search queries (it is told today's date)
2. **Search** — plain Python calls Tavily for each query in parallel (Brave if Tavily fails); duplicates removed, max 8 sources
3. **Extract** — the LLM pulls facts from the sources, each tagged `[n]`
4. **Gap check** — if something important is missing, one extra search round (max 2 rounds)
5. **Write** — report with inline `[n]` citations; the Sources list is appended by code, so URLs can't be invented
6. **Verify** — the critic checks claims against the cited sources; on failure the writer rewrites with its feedback

Credit protection: identical queries are cached for 6 hours and searches are capped at `SEARCH_DAILY_LIMIT` (default 40/day). If search is unavailable, the report falls back to model knowledge instead of failing.

Reports are reused from cache (1 hour) or long-term memory (`LTM_FRESH_HOURS`, default 24) unless **Fresh search** is ticked.

---

### API Endpoints

All requests need the header `X-API-Key: your-key` if you set one.

**Submit a research job:**
```bash
curl -X POST http://<alb_dns>/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"topic": "AI chip market 2025", "session_id": "abc123", "output_format": "text", "fresh": true}'
```
Returns: `{"job_id": "...", "session_id": "..."}`. `fresh` is optional (default `false`).

**Poll for result:**
```bash
curl http://<alb_dns>/result/<job_id> -H "X-API-Key: your-key"
```
Returns `{"status": "pending"}` until done, then the full report plus `sources` (id, title, url), `live_sources`, `sources_count`, `search_rounds` and `origin` (`pipeline` / `cache` / `memory`).

**Smoke test a deployment** (runs one fresh job and checks it has live sources and citations):
```bash
./scripts/smoke_test.sh <alb_dns> [your-api-key]
```

**Download as PDF:**
```bash
curl http://<alb_dns>/result/<job_id>/pdf -H "X-API-Key: your-key" -o report.pdf
```

**Get session history:**
```bash
curl http://<alb_dns>/session/<session_id> -H "X-API-Key: your-key"
```

**Get report diff (what changed vs previous):**
```bash
curl http://<alb_dns>/diff/<topic> -H "X-API-Key: your-key"
```

**Redis and system stats:**
```bash
curl http://<alb_dns>/stats -H "X-API-Key: your-key"
```

**Health check (no auth needed):**
```bash
curl http://<alb_dns>/health
```

---

## LangSmith — Traces and Evaluation

Every research job automatically:
1. Traces every agent node (search, summarize, write, verify) to LangSmith
2. Runs LLM-as-judge evaluations (relevance, completeness, hallucination risk, quality, and groundedness when the report has web sources)
3. Saves scores to a LangSmith dataset called `research-agent-reports`

View traces: https://smith.langchain.com → Project: `research-agent`

**Trigger batch evaluation manually** (runs the agent on recent user topics from the DB):
```bash
curl -X POST http://<alb_dns>/run-evaluation \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{}'
```

Pass specific topics instead:
```bash
curl -X POST http://<alb_dns>/run-evaluation \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"topics": ["quantum computing", "AI regulations"]}'
```

---

## PyRIT Red Team Dashboard

Open in browser:
```
http://<alb_dns>:8001/
```

This runs 4 types of attacks against your app to check if the guardrails are working:

| Attack | What it does |
|---|---|
| **Jailbreak** | Tries to bypass safety instructions directly |
| **XPIA** | Hides malicious instructions inside a research topic |
| **Crescendo** | Escalates from innocent questions toward harmful content step by step |
| **Skeleton Key** | Claims authority (researcher, CISO approval) to bypass restrictions |

Click **Run Selected Attacks** → wait 2–5 minutes → results appear showing BLOCKED or PASSED with a risk score.

Results are saved in Redis and survive container restarts.

**Run attacks via API:**
```bash
# All attack types
curl http://<alb_dns>:8001/run-attacks

# Specific types
curl "http://<alb_dns>:8001/run-attacks?types=jailbreak,xpia"

# Get results
curl http://<alb_dns>:8001/results
```

The weekly red team also runs automatically every Monday at 2am UTC via EventBridge.

---

## Pause / Resume (save credits)

Scaling ECS to 0 and stopping RDS is **not enough**: the ALB, Redis, VPC endpoints and public IPs still bill by the hour (~$3.70/day), auto-scaling puts the app back to 1 task, and the weekly red team still starts a task.

**Pause** (removes ALB, Redis, VPC endpoints, ECS services, auto-scaling, weekly red team; stops RDS):
```bash
./pause.sh        # Windows: pause.bat
```

**Resume** (recreates them from the `:latest` images in ECR, 5–10 min):
```bash
./resume.sh       # Windows: resume.bat
```

Kept while paused: RDS data, ECR images, Secrets Manager values, VPC, IAM, Bedrock Guardrail — a few cents a day.

- Redis is recreated empty, so cache, sessions and queued jobs are lost. Reports in Postgres survive.
- The ALB URL changes on every resume.
- AWS starts a stopped RDS instance again after 7 days — re-run pause if you stay paused longer.
- Pushing to `main` while paused still builds and pushes images; the ECS deploy is skipped.

---

## Tear Down Everything

```bash
cd terraform
terraform destroy -var="app_image=placeholder" -var="pyrit_image=placeholder"
```

#### Delete ECR Repo , S3 Bucket , Dynamo DB and Secret Manager

```bash
aws secretsmanager delete-secret --secret-id "research-agent/config" --force-delete-without-recovery --region us-east-1

```

Type `yes` when asked. This deletes all AWS resources — ECS, RDS, Redis, ALB, VPC, Bedrock Guardrail, Secrets Manager, ECR repos, everything.

> **Note:** RDS has deletion protection enabled. Terraform will remove it, but AWS will take a final snapshot first (named `research-agent-postgres-final-snapshot`). This is intentional so you don't lose data by accident.
