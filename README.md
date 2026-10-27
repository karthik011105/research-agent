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

## Prerequisites

Install these before starting:

| Tool | Install | Check |
|---|---|---|
| AWS CLI | https://aws.amazon.com/cli/ | `aws --version` |
| Terraform | https://developer.hashicorp.com/terraform/install | `terraform --version` |
| Git | https://git-scm.com/downloads | `git --version` |

Docker is **not needed** on your machine. GitHub Actions builds and pushes images automatically.

---

## Setup — Follow in Order

### 1. Configure AWS credentials

```bash
aws configure
```

Enter:
- **AWS Access Key ID** — AWS Console → your name (top right) → Security Credentials → Create access key
- **AWS Secret Access Key** — shown once at creation, copy it immediately
- **Default region** — `us-east-1`
- **Default output format** — `json`

---

### 2. Create the Terraform backend (one time only)

Terraform needs an S3 bucket and DynamoDB table to store its state. Run the bootstrap script to create them:

**Windows:**
```cmd
bootstrap.bat
```

**Mac / Linux / Git Bash:**
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```

Expected output:
```
S3 bucket  : research-agent-tfstate-<your-account-id>
DynamoDB   : research-agent-tf-locks
Bootstrap complete.
```

It also writes `terraform/backend.hcl` (git-ignored) with your account's bucket name. Re-run it whenever you switch AWS accounts.

---

### 3. Create a GitHub repo and add secrets

1. Go to https://github.com and create a new repo named `research-agent`

2. Push this project to it:
```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/research-agent.git
git push -u origin main
```

3. Add these two secrets (repo → Settings → Secrets and variables → Actions → New repository secret):

| Secret Name | Where to get it |
|---|---|
| `AWS_ACCESS_KEY_ID` | Same key you used in Step 1 |
| `AWS_SECRET_ACCESS_KEY` | Same key you used in Step 1 |

---

### 4. Deploy all AWS infrastructure

```bash
cd terraform
terraform init -reconfigure -backend-config=backend.hcl
terraform apply -var="paused=true"
```

`paused=true` creates only the parts that don't bill by the hour (VPC, RDS, ECR, Secrets, IAM, Guardrail). The ECR repos are still empty at this point, so there is nothing for ECS to run yet — Step 7 builds the images, then `resume` starts everything.

Type `yes` when asked. Takes 5–10 minutes.

This creates: VPC, subnets, ECS cluster, ALB, ElastiCache Redis, RDS PostgreSQL, Bedrock Guardrail, Secrets Manager, ECR repos, IAM roles, VPC endpoints, auto-scaling, EventBridge weekly red team schedule.

After it finishes, note these outputs — you'll need them:
```
alb_dns        = "research-agent-alb-xxxxxxx.us-east-1.elb.amazonaws.com"
app_ecr_url    = "123456789.dkr.ecr.us-east-1.amazonaws.com/research-agent-app"
pyrit_ecr_url  = "123456789.dkr.ecr.us-east-1.amazonaws.com/research-agent-pyrit"
```

---

### 5. Get your API keys

You need four keys (Brave is optional):

| Key | Where to get it |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `LANGSMITH_API_KEY` | https://smith.langchain.com → Profile → API Keys → Create |
| `TAVILY_API_KEY` | https://app.tavily.com → sign up → copy the key (`tvly-...`). Free: 1,000 searches/month |
| `BRAVE_API_KEY` *(optional)* | https://api-dashboard.search.brave.com → Subscriptions → pick a plan → API Keys. Only used if Tavily fails |

Without `TAVILY_API_KEY` the app still works, but reports come from model knowledge only (shown as **Model knowledge only** in the UI).

LangSmith is free. It traces every agent run and stores evaluation scores automatically — no extra setup needed after you add the key.

---

### 6. Fill in Secrets Manager

Terraform already filled in Redis URL, database URL, Guardrail ID, and all tuning parameters. You only need to add your API keys.

Go to: **AWS Console → Secrets Manager → `research-agent/config` → Retrieve secret value → Edit**

Replace the `REPLACE_ME` values:
```json
{
  "OPENAI_API_KEY":    "sk-...",
  "GROQ_API_KEY":      "gsk_...",
  "LANGSMITH_API_KEY": "ls__...",
  "TAVILY_API_KEY":    "tvly-..."
}
```

Save. Leave everything else as is. If a key is missing from the JSON (older installs), add it as a new line — mind the comma on the line before.

> Terraform never overwrites this secret after it is created, so your keys are safe across `terraform apply`. Running tasks read the secret only at startup — after editing it, restart the app: **ECS → research-agent-cluster → research-agent-app → Update service → Force new deployment**.

**Optional — set an API key to protect your endpoints:**

Add this field to the same secret:
```json
"API_KEY": "any-string-you-choose"
```

If set, every request to the app must include the header `X-API-Key: your-string`. The frontend has a field to enter it (saved in your browser). If left empty, the app runs without auth.

---

### 7. Build the images, then start the stack

The `git push` in Step 3 ran before the ECR repos existed, so trigger a build now: **GitHub repo → Actions → Build and Deploy → Run workflow** (or push any commit to `main`).

Wait for it to turn green (~5–10 minutes). It:
1. Runs the test suite
2. Builds the app, PyRIT, and TensorZero Docker images and pushes them to ECR
3. Skips the ECS deploy, because the stack is still paused

Then start everything (5–10 minutes):
```bash
./resume.sh       # Windows: resume.bat
```
It prints the app URL. From now on every push to `main` deploys automatically while the stack is running, and rolls back if the new version is unhealthy.

---

## Using the App

### Frontend

Open in browser:
```
http://<alb_dns>/
```

1. Enter your API key (if you set one in Step 6) — it saves in your browser
2. Type a research topic
3. Choose output format (text / PDF / JSON)
4. Tick **Fresh search** to skip cached/remembered reports and always search the live web
5. Click **Start Research** — polls automatically until done
6. The badge shows **Live web · N sources** or **Model knowledge only**; sources are listed under the report
7. Click **Show Changes vs Previous** to see what changed since last report on that topic

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
