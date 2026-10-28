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
