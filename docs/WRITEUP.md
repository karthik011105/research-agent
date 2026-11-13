# Where do multi-agent research systems break?
### Benchmarking accuracy, learned critic debate, and attack propagation in a live-web research agent

*Draft. The method is complete. Numbers marked **[pending]** are filled in by the first full run (see "Reproducing").*

---

## 1. Why this exists

Research agents are usually shown by example: a polished report, a list of sources. That shows they *work*. It doesn't show whether they're **better** than something simpler, **where** they go wrong, or **how much** that costs.

This project asks three measurable questions of one live system, at free-tier scale:

1. **Does the multi-agent architecture beat simpler systems on the same model?** If so, by how much, and at what cost?
2. **Do two critics that debate catch more errors than one, and when is the debate worth paying for?**
3. **When a web page is poisoned, which stage of the pipeline stops it? Can a poisoned report spread to other users through shared memory?**

## 2. The system

A topic is split by a **planner** into up to three sub-questions. Each sub-question gets the source kinds that fit it: web, news, research papers (OpenAlex, arXiv) or community discussion (Hacker News). LangGraph searches each sub-question in its own parallel branch. A **merge** step deduplicates the results and numbers sources once, so citations from different branches can't clash. It also scrapes thin pages (behind an SSRF guard) and drops blocked domains. **Extract** branches pull cited facts per sub-question. A **gap check** may add a search round. The **writer** produces a report with `[n]` citations, and the source list is appended by code, so URLs can't be invented.

Verification is done by a **critic panel**. The report is split into individual claims, and each claim is paired with the most relevant passages of its cited sources. A *fact-checker* (gpt-oss-120b) and a *skeptic* (Qwen, a different model family) give a verdict on every claim. Each must **quote** the source. Code discards any vote whose quote isn't really in the source, so a critic can't win an argument with an invented quote. Disagreements go to a **debate**: each critic sees the other's verdict and quote and may revise. The writer then gets an exact list of claims to fix or remove.

The stack: Groq through a TensorZero gateway, on AWS (ECS Fargate, RDS Postgres + pgvector, ElastiCache Redis, Bedrock Guardrails), built with Terraform and deployed by CI.

## 3. Experiment 1: does the architecture help?

**Systems.** All three write with the same model (gpt-oss-120b) and the same writer prompt:
- **A:** a single call, no search
- **B:** basic RAG, meaning one web search and one call
- **C:** the full multi-agent system

**Fairness controls:**
- **Saved searches.** Search results are snapshotted per topic and replayed, so B and C see identical sources and re-runs cost no searches.
- **Pinned writer.** Free-tier rate limits normally make the gateway fall back to a smaller model silently. Benchmark runs pin a writer variant with no fallback. When the daily quota runs out, a run stops instead of quietly switching models. The model that served each call is recorded.
- **Topics.** There are 25 topics: news, technical, comparisons, and **trap topics** about things that don't exist (a "2026 Nobel Prize in Mathematics", a fictional city). The free-tier run uses an 8-topic subset, two per category.

**Scoring.**
- **By code:**
  - citation validity
  - the share of numeric claims that carry a citation
  - report structure
  - AI calls, tokens, searches and latency
- **By two judges:**
  - The judges come from different model families (Qwen and gpt-oss) and are averaged.
  - Each judge sees the report's own cited sources first, then the topic's evidence pool.
  - Runs where the judges are 3 or more points apart are flagged.
  - Because every system writes with the same model, any self-preference of the gpt-oss judge affects all systems equally.

**Lessons from the trial runs, which changed the method before any result was trusted:**
- The first judge saw only 500 characters per source, so it marked correctly cited facts as "fabricated" (accuracy 2/10 for a report that was right). Giving the judge each report's own cited sources moved the same report to 9/10.
- A single judge contradicted itself across runs on identical evidence. That's why there are two judges with disagreement flags.

**Results [pending].**

| | A · single call | B · basic RAG | C · multi-agent |
|---|---|---|---|
| Accuracy (1–10) | | | |
| Unsupported claims per report | | | |
| Citation validity | – | | |
| Honesty on trap topics (1–10) | | | |
| AI calls · tokens · seconds | | | |

One early observation, an anecdote and not a result: on the fictional city, A invented a complete economic profile (honesty 1/10). On the non-existent Nobel Prize, C stated it doesn't exist and reported the real 2026 mathematics prizes from its sources (10/10).

## 4. Experiment 2: do debating critics help, and when?

**Labels for free.** Test claims are built from real source sentences:
- **True claims:** kept verbatim, so they're supported by construction.
- **False claims:** copies broken in one of four ways:
  - a number changed
  - a year shifted
  - a named organisation swapped for one from another source
  - two sources merged into one claim

Every critic setup judges the same claims, so catch rate and false-alarm rate need no manual labelling.

**Setups:**
- one YES/NO critic
- two critics without debate (a remaining tie goes to the stricter verdict)
- two critics with debate

**Learning (Experiment 2b).** Debate costs AI calls, so the system learns **when** it's worth it.
- A contextual bandit chooses 0, 1 or 2 debate rounds per claim type.
- Reward: a correct verdict, minus the cost of each round.
- Because a full two-round debate also reveals what 0 and 1 rounds would have decided, one labelled example updates every option. This is full-information learning, which suits a small budget.
- Separately, **trust weights** (multiplicative weights) learn which critic to believe for numbers, dates, attributions and general claims. They replace the fixed "stricter wins" tie-break.

**Results [pending].**

| Critic setup | Errors caught | False alarms | Extra calls |
|---|---|---|---|
| One critic | | | |
| Two critics | | | |
| Two critics + debate | | | |
| Learned debate policy + trust | | | |

Caught rate by error type (number / date / attribution / merge): **[pending]**.

## 5. Experiment 3: where do attacks stop?

**Indirect prompt injection.** One poisoned page is inserted at the top of real search results. This is where an attacker optimising for ranking would place it. Five attack types are tested:
- a hidden instruction to state a false statistic
- fake system/developer markup
- a request to reveal the system prompt
- a demand to be cited for every claim
- plain misinformation with no instructions

Each comes in three phrasings, and each page carries a **unique made-up organisation name** (a canary).

**Tracing.** The pipeline streams every node's output, and we record whether the canary reached:

`retrieved → extracted → drafted → critic → final report → output guardrail`

The first stage where it disappears is the defense that worked.

**Results [pending]:** a matrix of attack type × stage reached, for the single critic and for the debate panel.

**Memory poisoning.** A poisoned report is planted in the shared cache and long-term memory exactly as the pipeline would store it. Then a *different session* asks:
- a paraphrase of the same topic (cache or memory hit)
- a related topic (memory used as background by the writer)
- an unrelated topic (control)

Everything planted is deleted afterwards.

**Defenses compared:** none · store only critic-approved reports · per-session memory · both.

The design already exposes a structural point. Storing only verified reports protects other users **only if the critic caught the poison**. Per-session memory blocks cross-user reuse **regardless**. The live run measures how often each case occurs. **[pending]**

**Self-healing guardrails.** When an attack gets through, an LLM proposes a regular expression for the *technique*, not the wording. Code accepts it only if all of these hold:
1. It catches at least 80% of the attack's **held-out phrasings**, which the proposer never saw.
2. It flags at most 2% of **real benign pages** (the benchmark's saved sources).
3. It has no unbounded nested quantifiers.

The owner approves a rule before it filters pages. A rule written for one exact phrasing fails test 1, and an overly broad one fails test 2. Both behaviours are covered by tests. Rule acceptance rate and the attack matrix with approved rules: **[pending]**.

## 6. Learning from scores in production

A Thompson-sampling bandit chooses the writer variant per report:
- the default prompt
- a smaller model
- a stricter "cite every number" prompt

The reward is mostly objective: the share of claims the critic panel kept, citation validity, whether the first draft was accepted, and latency. The user's 👍/👎 is added on top.

**Safeguards:**
- A 10% exploration floor.
- A kill switch.
- Benchmark runs never use the bandit.

In simulation the bandit routes over 70% of late traffic to the best variant, with low regret. Live convergence: **[pending]**.

## 7. Limitations

- **Free-tier scale.** About 200k tokens/day on the main model is roughly 4 full multi-agent reports a day, which is why the benchmark uses 8 topics. Differences smaller than about 1 point of judge accuracy shouldn't be read as real.
- **LLM judges are noisy.** Two judges, objective metrics and error-injected labels reduce the noise but don't remove it. Results are reported with judge agreement.
- **Free data sources.** arXiv, Semantic Scholar and GDELT often rate-limit cloud IPs. OpenAlex is the reliable source for papers.
- **Attack set.** The attacks are templates written for this study (five types × three phrasings), not a broad adversarial corpus. The goal is to locate which stage stops what, not to claim robustness.

## 8. Reproducing

```bash
bash scripts/resume_safe.sh
python bench/run.py --url <app> --label v1 --set small          # Experiment 1 (resumable across days)
# Experiment 2: POST /bench/critic-eval  {label, topic_id, mode: single|panel|debate, learn: true}
# Experiment 3: POST /security/attack-run, /security/memory-run, /security/rules/propose
python bench/report.py --url <app> --label v1 --set small       # tables -> bench/results/v1.md
bash scripts/pause_safe.sh
```

All experiment code is unit-tested (369 tests), including end-to-end pipeline runs with scripted models, attack-tracing scenarios and bandit convergence.
