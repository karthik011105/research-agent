# Roadmap: from "it works" to "it's measurably better, self-improving and safer"

**Status as of 2026-11-13:** Phase 1 is live (commit `05896b0`).
- Multi-source research over parallel sub-questions
- Scraper
- Question router
- Guardrail v2
- Job hand-back on shutdown
- 84 tests

## Overview

| Step | What | Type | Sessions | Result you can claim |
|---|---|---|---|---|
| 0 | Housekeeping and a local dev stack | Setup | ½ | Development at near-zero cloud cost |
| 1 | Foundation fixes | Fixes | 1 | Faster cache, honest red-team numbers |
| 2 | **Benchmark** ⭐ (25 topics × 3 systems) | Measurement | 2–3 | "Multi-agent beats single-pass by X% at Y× the cost" |
| 3 | **Critic panel + debate** ⭐ | Rule-based | 2–3 | "Debate catches X% more misattributed claims" (or doesn't, also a finding) |
| 4 | **Learning from scores** ⭐: critic trust, learned debate, writer choice | Learning-based | 3–4 | "The system learns when to debate and which model to use" |
| 5 | **Security research** ⭐: injection, propagation, memory poisoning, self-healing | Research | 3–4 | "Where multi-agent systems break, and how we fixed it" |
| 6 | Realtime UX: live progress and topic monitoring | Feature | 1–2 | A realtime demo |
| 7 | Publish: README, write-up, video | Delivery | 1–2 | A portfolio-ready project |

**Total: about 15–20 sessions.** Each step builds on the one before. Step 2 measures everything that follows, and the rule-based Step 3 is the baseline that the learning in Step 4 must beat.

## Working rules for every step
- Runs happen on **AWS** (owner's choice). Code is written and unit-tested on the laptop with plain Python (no Docker). The stack is resumed only to deploy, verify, run the benchmark or demo, and paused straight after, at about $0.27/hour while running.
- `docker-compose.yml` stays in the repo as an optional local alternative, but is not part of the workflow.
- Every change ships with tests and CI must be green before deploying. After a deploy, verify live once the old task has stopped. The drain now takes 30 s.
- Changes that loosen safety, and every `terraform apply`, need the owner's explicit OK.
- Free-tier budgets:
  - Groq: about 1,000 requests/day per model and 8k tokens/min.
  - Tavily: 1,000 searches/month.
  - Long runs are spread over days, and search results are saved and replayed.

---

## Step 0: Housekeeping (½ session)
- [x] Pause the AWS stack whenever it isn't being used.
- [x] Pin `tensorzero/gateway` to an exact version (2026.6.0 + digest), not `:latest`, so a new release can't change behaviour on the next deploy.
- [x] Optional local stack (`docker-compose.yml`, `CONFIG_SOURCE=env`, `scripts/local_env.py`). It isn't used, because runs happen on AWS.
- [x] CPU-only PyTorch in the app image: drops about 3 GB of unused CUDA libraries, so builds, deploys and ECR storage are smaller.

**Done when:** the next deploy builds with the pinned gateway and the slimmer image, and a live job succeeds.
✅ **Done 2026-11-13** (`981d16d`): the app image is 628 MB compressed, and a live job succeeded.

## Step 1: Foundation fixes (1 session)
- [x] Semantic cache: stop scanning every key per lookup. It now uses one Redis hash index plus a matrix product, sha256 keys and a bounded size.
- [x] Load the embedding model once and share it (`app/embeddings.py`, run off the event loop).
- [x] Related-memory lookup takes the most similar report, not the most recent (the root cause of the AI-chips leak).
- [x] Red-team scoring honesty: the runner never sent the API key, so every attack counted as "blocked" on a 401. Outcomes are now blocked / refused / complied / error, and the dashboard's XSS is fixed.
- [x] Configurable blocklist of low-quality source domains (`BLOCKED_DOMAINS`, prediction markets by default).
- [x] Static import check in CI. A removed name crashed the app at startup, and the slim CI dependencies hid it.

**Done when:** tests cover each fix, and red-team numbers distinguish blocked from errors.
✅ **Done 2026-11-13**:
- 170 tests pass.
- Live checks:
  - all 4 source types used
  - the new cache answers in 3 s
  - red team: 4/4 jailbreaks blocked by the input guardrail, 0 errors

## Step 2: Benchmark ⭐ (2–3 sessions). The measuring tool for every later step
- [ ] `bench/topics.yaml` holds 25 fixed topics:
  - current news
  - technical research
  - comparisons
  - 3 "trap" topics whose honest answer is "not known / not found"
- [ ] **Search snapshots**: each topic's search results are saved once and replayed, so every system sees identical sources, results are reproducible, and Tavily is spent only once.
- [ ] **Systems compared**, all on the same model (`gpt-oss-120b`), so the benchmark measures the architecture rather than the model:
  - **A** single pass, no search
  - **B** basic RAG (one search, one LLM call)
  - **C** the full multi-agent pipeline
- [ ] **Metrics**:
  - *Objective, computed by code:* citation validity (every `[n]` exists and is relevant), coverage of sub-questions, tokens, searches, latency.
  - *Judged:* factual accuracy, hallucination, completeness. The judge is a different model from the writer (to avoid self-preference), plus a human spot-check of 5 topics.
- [ ] `bench/run.py` is resumable (it stops and continues within free-tier limits) and writes `bench/results/*.json`. `bench/report.py` produces a table and chart.

**Done when:** one command reproduces the A/B/C table.

**Status (2026-11-13):**
- Built and deployed.
- Owner's choice: free Groq tier with the 8-topic `small` set, about 2 days, because gpt-oss-120b has a 200k tokens/day limit.
- What the trial runs taught us:
  - The judge needs each report's own cited sources.
  - Two judges are used (Qwen + gpt-oss), with disagreements flagged.
  - Benchmark runs pin a no-fallback writer, so the quota running out pauses the run instead of mixing models.
  - Qwen can't serve as a writer fallback (1k output tokens/min).
- `v1` stopped on the daily quota. Continue the next day with `bash scripts/resume_safe.sh`, then `python bench/run.py --url <alb> --label v1 --set small`.

## Step 3: Critic panel with debate (2–3 sessions). Rule-based, and the baseline for Step 4
- [ ] **3a Claim splitter**:
  - Break the report into single claims, each tied to its `[n]` source.
  - Pick the most relevant passages of that source by embedding similarity, batched to fit token limits.
- [ ] **3b Two critics** with different roles **and different models**:
  - *fact-checker* (gpt-oss-120b): is the claim stated in the source?
  - *skeptic* (qwen): is it misattributed, outdated, exaggerated, or merged from two sources?
  - Every verdict must **quote** the source text it relies on.
- [ ] **3c Debate**: only for claims the critics disagree on. Each critic sees the other's verdict and quote and may revise (maximum 2 rounds).
- [ ] **Judge** (code, not another LLM):
  - Quotes are verified to exist in the source text; a verdict with a fabricated quote is discarded.
  - A remaining tie goes to the stricter verdict. Step 4a replaces this rule with learned trust.
- [ ] **Final decision per claim** (supported / fix / remove), with exact fixes sent to the writer. Covers cases like ISO 42001 clauses presented as EU law.
- [ ] **Error injection: labelled data for free**:
  - Copy correct claims and break them in known ways: change a number, swap who said it, shift a date, merge two sources.
  - Every verdict can then be scored right or wrong automatically. Steps 3 and 4 use this data.
- [ ] **Compare four setups** (errors caught, false alarms, extra calls, latency):
  1. no critic
  2. single critic
  3. panel without debate
  4. panel + debate

**Done when:**
- The four-setup table exists, whichever setup wins.
- The best rule-based setup is the default.
- A regression test covers the ISO-vs-EU case.

**Status (2026-11-13): built, not yet measured live.**
- Code:
  - `app/critics.py`: claim splitter, passage picker, fact-checker + skeptic, quote check, debate, code judge.
  - `app/critic_eval.py`: error injection and scoring.
  - `POST /bench/critic-eval` runs one setup on a topic's saved sources.
- `CRITIC_MODE` stays `single` in production until the comparison is run (`single` / `panel` / `debate`).
- The live comparison needs the benchmark's saved sources and Groq quota, so it runs together with Step 2.

## Step 4: Learning from scores ⭐ (3–4 sessions). The system improves itself from feedback
All three parts are **learning-based**. No model weights change; the system learns *decisions* from outcomes. Statistics are stored in the existing Postgres, so nothing is added to the bill. TensorZero's built-in experiments would need a ClickHouse database, an extra always-on cost, so they aren't used.

**Order: critics first (4a, 4b), then the writer (4c).** The writer's reward comes from the critics' verdicts, so the critics must be reliable before the writer learns from them.

### 4a: Critic trust scores (online learning, multiplicative weights)
- [ ] Each critic keeps a reliability weight **per claim type** (number, date, attribution, general). All weights start equal.
- [ ] After each labelled verdict (error injection, or the code-verified quote check), the weight of the critic that was right goes up and the other's goes down.
- [ ] The judge settles disagreements by weighted vote instead of "stricter wins".

**Done when:** weighted voting beats the fixed tie-break rule on the error-injection set, and a chart shows how the trust weights evolve.

### 4b: Learned debate policy ⭐ (contextual bandit)
- [ ] For each claim, the policy chooses: accept the panel verdict, debate 1 round, or debate 2 rounds.
- [ ] Context: whether the critics disagree, their confidence, and the claim type.
- [ ] Reward: +1 for a correct final verdict, minus a cost per extra LLM call.
- [ ] Exploration floor plus a kill switch that reverts to the rule-based policy.

**Done when:** a cost/accuracy chart compares the learned policy with "never debate" and "always debate", e.g. "keeps 95% of always-debate accuracy at 40% of the calls".

### 4c: Writer variant selection (Thompson-sampling bandit)
- [ ] **Variants**:
  - `A` = gpt-oss-120b, current prompt
  - `B` = qwen, current prompt
  - `C` = gpt-oss-120b, "every number cited" prompt

  TensorZero runs each call with the chosen variant pinned. Extract and planner can follow later.
- [ ] **Reward per job** (0–1), mostly from objective signals:
  - claim pass rate from the critic panel
  - citation validity
  - whether the first draft was accepted
  - user 👍/👎 when given (new UI button)
  - small cost/latency penalty
- [ ] **Bandit**: a Beta posterior per variant, sampled per job, with each variant kept at ≥ 10% of jobs and a kill switch.
- [ ] **Storage**: `job_variants` (job → variant, reward parts) and `variant_stats` (posteriors), in Postgres.
- [ ] **Warm start**: run each variant over the benchmark topics, replaying the search snapshots, then keep learning from live traffic.
- [ ] **Visibility**: a `/stats/learning` endpoint and a UI chart of each variant's win probability and share of traffic over time.
- [ ] **Honesty checks**:
  - A replay simulation shows convergence, measured as cumulative regret.
  - The benchmark is re-run with the learned choice against the old fixed choice.

**Done when:** the chart shows traffic shifting to the best variant, and the benchmark confirms that variant really scores higher.

**Status (2026-11-13): built, not yet measured live.**
- `app/learning.py`: CriticTrust (4a), DebatePolicy (4b), WriterBandit (4c), LearningService, with state in Postgres.
- The critic panel uses learned trust and debate rounds. Writer variants B (`writer_20b`) and C (`cite_numbers`) are weight-0 TensorZero variants pinned by the bandit.
- Rewards are computed after each job, plus 👍/👎 (`POST /feedback/{job_id}`). The learners are visible at `GET /stats/learning`.
- `LEARNING_ENABLED` defaults to false: production stays rule-based until learning is measured. `POST /bench/critic-eval` with `learn=true` trains the critics from labelled claims either way.
- Simulations in the tests show convergence:
  - the bandit sends over 70% of late traffic to the best writer
  - the debate policy learns 1 round for numbers and 0 for general claims
  - trust shifts towards the critic that is right
- Live measurement runs together with Steps 2 and 3, once Groq quota allows.

**Stretch goal, after Step 7:**
- Train a small open critic model with real RL (DPO/GRPO) on the labelled verdicts.
- This needs a GPU (e.g. Colab) and its own model host, because Groq can't serve custom weights.

## Step 5: Security research ⭐ (3–4 sessions). Where multi-agent systems break
- [ ] **Real indirect prompt injection (XPIA)**: a test search provider serves poisoned web pages. Today's "XPIA" prompts only arrive through the topic box.
- [ ] **Attack propagation**:
  - Each poisoned page carries a unique canary.
  - The study traces whether it reaches extract facts → writer report → critic panel → output guardrail.
  - It also tests whether a poisoned source fools one critic but not both.
  - Output: a matrix of which stage stops which attack type.
- [ ] **Memory poisoning**: plant a false report in long-term memory and the shared cache (test environment), then ask related and unrelated topics as another user. Measure the contamination rate.
- [ ] **Defenses, then re-measure**:
  - memory provenance and verification before reuse
  - per-user memory
  - verified writes
  - Report the contamination rate before and after.
- [ ] **Self-healing guardrails**:
  - When an attack passes, an LLM proposes a candidate rule.
  - The rule is tested against the attack set **and** a harmless-question set, to catch false positives like the misconduct filter.
  - The owner approves before it is deployed.

**Done when:** there is a propagation matrix, contamination numbers before and after the defenses, and one full self-healing cycle demonstrated.

**Status (2026-11-13): built, not yet run live.** Modules in `app/security/`:
- `attacks.py`: 5 attack kinds × 3 phrasings, each carrying a unique canary.
- `propagation.py`: stage tracing through LangGraph `astream`.
- `memory_poisoning.py`: plant a poisoned report, query as another user, then clean up.
- `rules.py`: rules are proposed by an LLM and then tested by code: at least 80% recall on held-out phrasings, at most 2% false positives on real saved pages, and no catastrophic regexes. Only owner-approved rules are applied, and they remove matching sentences from pages before any LLM reads them.

Endpoints:
- `/security/attack-run`
- `/security/memory-run`
- `/security/rules/propose`
- `/security/rules`
- `/security/rules/{id}/approve|reject`

New defenses, off until measured:
- `MEMORY_SCOPE=session`
- `MEMORY_VERIFIED_ONLY=true`

Already visible in the tests: verified-only writes help only when the critic caught the poison, while session scope blocks cross-user reuse either way.

## Step 6: Realtime UX (1–2 sessions)
- [ ] **SSE progress stream**: plan → searches per sub-question → sources read → writing → critic panel (debates as they happen), plus a 🛡 event whenever a defense fires.
- [ ] **Topic monitoring**: a saved topic is re-researched on a schedule (EventBridge), with a "what changed" diff.

**Done when:** the UI shows live progress, and a monitored topic produces a dated diff.

**Status (2026-11-13): built, not yet run live.**
- 6a: `app/progress.py`.
  - Every agent emits events into a Redis stream per job.
  - `GET /stream/{job_id}` relays them as SSE, replays earlier events, and sends a keep-alive every 15 s for the ALB.
  - The UI shows a live timeline (read with fetch, because EventSource can't send the API key), with 🛡 lines when a defense fires.
- 6b: `app/monitoring.py`.
  - An in-app scheduler (no extra AWS service) uses a Redis lock per time slot, so a topic runs once even with several tasks.
  - The "what changed" diff of claims is computed by code, with no LLM tokens.
  - At most 5 topics, at least every 6 hours apart. `/monitors` endpoints and a UI section.

## Step 7: Publish (1–2 sessions)
- [x] README rewritten with pipeline, architecture and defense-in-depth diagrams (2026-11-13). The results tables are placeholders until the live run.
- [ ] Fill in the README and write-up results from the live run.
- [x] Write-up drafted: `docs/WRITEUP.md`. The method is complete and the results are pending.
- [ ] A 2-minute demo video and a stable URL (free DuckDNS).

---

**Not doing, for now:**
- MCP server
- Neo4j knowledge graph
- React rewrite
- human-in-the-loop plan approval
- training a model with real RL (stretch goal)

**Known limits to state honestly:**
- arXiv, Semantic Scholar and GDELT often rate-limit requests from AWS; OpenAlex is the reliable source for papers.
- The judges are LLMs, so the benchmark uses objective metrics, error-injection labels and human spot-checks wherever possible.
