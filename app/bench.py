"""Benchmark: the same topics through three systems on the same model, scored by code and by a judge.

  single  one LLM call, no search                     - what the model knows on its own
  rag     one web search + one LLM call               - a basic retrieval-augmented baseline
  multi   the full multi-agent pipeline               - planner, parallel sub-questions, critic, ...

Fairness: every system writes with the same TensorZero function (report_write, gpt-oss-120b first), and search
results are snapshotted per topic, so re-runs and different systems see identical sources and don't spend the
Tavily budget again. Two judges from different model families (Qwen and gpt-oss) score each report and are
averaged; they check it against the report's own cited sources and the topic's evidence pool (every snapshotted
source), so even the no-search system is judged against the same facts.
"""
import json
import re
import time
import uuid
from datetime import date, datetime
from typing import Protocol
import redis.asyncio as aioredis
from app import metering
from app.agents import (
    _tz_call, build_graph, cited_ids, initial_state, normalize_citations, parse_json_object, sources_section,
)
from app.config import Config
from app.tools.search import format_sources, merge_sources, search_kind

SYSTEMS = ("single", "rag", "multi")
_SECTIONS = ("executive summary", "key findings", "analysis", "conclusion")
_EVIDENCE_CHARS = 13000        # ~3.3k tokens: report + evidence + answer stay under Groq's 8k tokens/min
_CITED_BUDGET = 11000          # every cited source must be visible to the judge, however many a report cites
_CITED_SOURCE_CHARS = 1500     # the writer's own per-source limit (source_char_limit)
_MIN_CITED_CHARS = 600
_POOL_SOURCE_CHARS = 450
_REPORT_CHARS = 7000
_DISAGREE_POINTS = 3           # judges further apart than this on accuracy are flagged
JUDGES = (("qwen", "research_judge"), ("gpt_oss", "research_judge_alt"))
MAIN_MODEL = "openai/gpt-oss-120b"     # the writer every system is meant to use


# ── storage ──────────────────────────────────────────────────────────────────

class BenchStore(Protocol):
    async def get_snapshot(self, topic_id: str, query: str, kind: str) -> list[dict] | None: ...
    async def put_snapshot(self, topic_id: str, query: str, kind: str, results: list[dict]) -> None: ...
    async def evidence(self, topic_id: str) -> list[dict]: ...
    async def save_run(self, run: dict) -> None: ...


class PostgresBenchStore:
    def __init__(self, pool):
        self.pool = pool

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bench_snapshots (
                    topic_id TEXT NOT NULL, query TEXT NOT NULL, kind TEXT NOT NULL,
                    results JSONB NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (topic_id, query, kind))""")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bench_runs (
                    id TEXT PRIMARY KEY, label TEXT NOT NULL, topic_id TEXT NOT NULL, system TEXT NOT NULL,
                    topic TEXT NOT NULL, status TEXT NOT NULL, report TEXT, sources JSONB, metrics JSONB,
                    judge JSONB, error TEXT, started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS bench_runs_label_idx ON bench_runs (label)")

    async def get_snapshot(self, topic_id, query, kind):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT results FROM bench_snapshots WHERE topic_id=$1 AND query=$2 AND kind=$3", topic_id, query, kind)
        return json.loads(row["results"]) if row else None

    async def put_snapshot(self, topic_id, query, kind, results):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO bench_snapshots (topic_id, query, kind, results) VALUES ($1,$2,$3,$4::jsonb) "
                "ON CONFLICT DO NOTHING", topic_id, query, kind, json.dumps(results))

    async def evidence(self, topic_id):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT results FROM bench_snapshots WHERE topic_id=$1 ORDER BY created_at", topic_id)
        return [s for r in rows for s in json.loads(r["results"])]

    async def all_evidence(self, limit: int = 300) -> list[dict]:
        """Saved search results across all topics (real, harmless web pages for testing filter rules)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT results FROM bench_snapshots ORDER BY created_at DESC LIMIT $1", limit)
        return [s for r in rows for s in json.loads(r["results"])]

    async def save_run(self, run):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO bench_runs (id, label, topic_id, system, topic, status, report, sources, metrics, judge,
                                        error, started_at, finished_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb,$11,$12,$13)
                ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, report=EXCLUDED.report,
                    sources=EXCLUDED.sources, metrics=EXCLUDED.metrics, judge=EXCLUDED.judge,
                    error=EXCLUDED.error, finished_at=EXCLUDED.finished_at""",
                run["id"], run["label"], run["topic_id"], run["system"], run["topic"], run["status"],
                run.get("report"), json.dumps(run.get("sources") or []), json.dumps(run.get("metrics") or {}),
                json.dumps(run.get("judge") or {}), run.get("error"), run["started_at"], run.get("finished_at"))

    async def done_runs(self, label: str, systems: list[str]) -> list[dict]:
        """Finished runs with every column, ready to be saved back (for re-judging)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM bench_runs WHERE label=$1 AND status='done' AND system = ANY($2) "
                                    "ORDER BY started_at", label, systems)
        out = []
        for r in rows:
            d = dict(r)
            for k in ("metrics", "judge", "sources"):
                if isinstance(d.get(k), str):
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    async def supersede(self, label: str, system: str, topic_ids: list[str]) -> int:
        """Retires finished runs so the runner does them again (e.g. after a fix to that system). Kept, not deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("UPDATE bench_runs SET status='superseded' WHERE label=$1 AND system=$2 "
                                        "AND topic_id = ANY($3) AND status IN ('done', 'error')", label, system, topic_ids)
        return int(result.split()[-1])

    async def list_runs(self, label: str, full: bool = False) -> list[dict]:
        cols = "*" if full else "id, label, topic_id, system, status, metrics, judge, error, started_at, finished_at"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT {cols} FROM bench_runs WHERE label=$1 ORDER BY started_at", label)
        out = []
        for r in rows:
            d = dict(r)
            for k in ("metrics", "judge", "sources"):
                if isinstance(d.get(k), str):
                    d[k] = json.loads(d[k])
            for k in ("started_at", "finished_at"):
                if d.get(k):
                    d[k] = d[k].isoformat()
            out.append(d)
        return out


def snapshot_search(store: BenchStore, topic_id: str):
    """search_kind that replays a topic's saved results for the same (query, kind), recording new ones."""
    async def search(redis, config, query, kind, allow_paid=True):
        key = " ".join(query.lower().split())
        saved = await store.get_snapshot(topic_id, key, kind)
        if saved is not None:
            return saved
        results = await search_kind(redis, config, query, kind, allow_paid=allow_paid)
        await store.put_snapshot(topic_id, key, kind, results)
        return results
    return search


# ── the three systems ────────────────────────────────────────────────────────

def _report_rules() -> str:
    return "Include: Executive Summary, Key Findings, Analysis, and Conclusion."


async def run_single(config: Config, topic: str) -> tuple[str, list[dict]]:
    report = await _tz_call(
        config, "report_write",
        f"Today's date: {date.today().isoformat()}\n"
        f"Write a comprehensive, well-structured research report on: '{topic}'\n"
        f"Use only your own knowledge (no sources are provided). Say so briefly in the Executive Summary, and say "
        f"when information may be out of date or when you don't know.\n{_report_rules()}",
    )
    return normalize_citations(report), []


async def run_rag(config: Config, redis: aioredis.Redis, topic: str, search) -> tuple[str, list[dict]]:
    sources = merge_sources([await search(redis, config, topic, "web")], [], config)
    report = normalize_citations(await _tz_call(
        config, "report_write",
        f"Today's date: {date.today().isoformat()}\n"
        f"Write a comprehensive, well-structured research report on: '{topic}', using the sources below.\n"
        f"Cite every claim that uses a source with its id in plain square brackets, like [3] or [1, 4]. "
        f"Do not add a sources or references list - it is appended automatically.\n{_report_rules()}\n\n"
        f"{format_sources(sources)}",
    ))
    return report + sources_section(report, sources), sources


async def run_multi(config: Config, redis: aioredis.Redis, topic: str, search) -> tuple[str, list[dict]]:
    final = await build_graph(config, redis, search_fn=search).ainvoke(initial_state(topic, "bench", [], ""))
    return final["report"], (final["sources"] if final["live_sources"] else [])


# ── scoring ──────────────────────────────────────────────────────────────────

def _body(report: str) -> str:
    return re.split(r"\n#+\s*Sources\s*\n", report)[0]


def report_metrics(report: str, sources: list[dict]) -> dict:
    """Objective, code-computed measures (no LLM)."""
    body = _body(report)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.strip()) > 20]
    numeric = [s for s in sentences if re.search(r"\d", re.sub(r"\[\d+(?:,\s*\d+)*\]", "", s))]
    cited = cited_ids(body)
    valid_ids = {s["id"] for s in sources}
    lower = body.lower()
    return {
        "words": len(body.split()),
        "sources": len(sources),
        "citations": len(cited),
        # share of [n] markers that point at a real source (None when the system has no sources)
        "citation_validity": round(len(cited & valid_ids) / len(cited), 3) if cited and sources else None,
        # share of sentences stating numbers/dates that carry a citation
        "numeric_claims": len(numeric),
        "numeric_cited_ratio": round(sum(1 for s in numeric if re.search(r"\[\d+", s)) / len(numeric), 3)
        if numeric else None,
        "sections_present": sum(1 for sec in _SECTIONS if sec in lower),
    }


def evidence_block(evidence: list[dict], cited: list[dict] | None = None) -> str:
    """Evidence for the judge, trimmed to fit free-tier token limits.

    The report's own cited sources come first with most of their text: in the first trial the judge saw only
    500 characters per source, so facts a report cited correctly from further down a page were marked
    "fabricated". All cited sources share one budget, so a report citing 12 sources is judged on all 12 - with a
    fixed 1,500 characters each, only the first ~6 fit, and facts from the rest looked invented (v2 run). The rest
    of the topic's evidence pool follows with shorter excerpts."""
    cited = [s for s in cited or [] if s.get("content")]
    per_cited = (max(_MIN_CITED_CHARS, min(_CITED_SOURCE_CHARS, _CITED_BUDGET // len(cited))) if cited
                 else _CITED_SOURCE_CHARS)
    seen, parts, used = set(), [], 0
    for items, chars in ((cited, per_cited), (evidence, _POOL_SOURCE_CHARS)):
        for s in items:
            url = (s.get("url") or "").rstrip("/")
            if url in seen or not s.get("content"):
                continue
            seen.add(url)
            piece = f"- {s.get('title', '')} ({s.get('published_date') or 'undated'}): {s['content'][:chars]}"
            if used + len(piece) > _EVIDENCE_CHARS:
                return "\n".join(parts)
            parts.append(piece)
            used += len(piece)
    return "\n".join(parts) or "(no evidence was retrieved for this topic)"


def _clamp(value, low, high):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return None


def _judge_prompt(topic: str, report: str, evidence_text: str, trap: bool) -> str:
    trap_rule = (
        'This is a TRAP topic: it asks about something that does not exist or is not known. Score "honesty" 1-10: '
        "10 = the report clearly says so, 1 = it invents a confident answer.\n" if trap else "")
    return (
        f"Evaluate this research report on '{topic}' against the EVIDENCE (web sources retrieved for this topic) "
        f"and well-established facts. A claim is supported if the evidence or well-established fact backs it; "
        f"do not penalise a claim only because the evidence is silent if it is common knowledge.\n{trap_rule}"
        f'Reply with JSON only: {{"accuracy": 1-10, "completeness": 1-10, "unsupported_claims": <number of '
        f'specific claims (numbers, dates, names, quotes) not backed by the evidence or well-established fact>, '
        f'"honesty": 1-10 or null, "reason": "<one sentence>"}}\n\n'
        f"EVIDENCE:\n{evidence_text}\n\nREPORT:\n{_body(report)[:_REPORT_CHARS]}"
    )


def _parse_verdict(reply: str, trap: bool) -> dict:
    parsed = parse_json_object(reply)
    unsupported = parsed.get("unsupported_claims")
    return {
        "accuracy": _clamp(parsed.get("accuracy"), 1, 10),
        "completeness": _clamp(parsed.get("completeness"), 1, 10),
        "unsupported_claims": int(unsupported) if isinstance(unsupported, (int, float)) else None,
        "honesty": _clamp(parsed.get("honesty"), 1, 10) if trap else None,
        "reason": str(parsed.get("reason", ""))[:300],
        "parsed": bool(parsed),
    }


async def judge_report(config: Config, topic: str, report: str, evidence: list[dict], trap: bool,
                       cited_sources: list[dict] | None = None) -> dict:
    """Two judges from different model families (see tensorzero.toml), called one after the other to stay inside
    per-minute token limits, then averaged. Large disagreements are flagged rather than hidden."""
    prompt = _judge_prompt(topic, report, evidence_block(evidence, cited_sources), trap)
    verdicts = {}
    for name, function in JUDGES:
        try:
            verdicts[name] = _parse_verdict(await _tz_call(config, function, prompt), trap)
        except Exception as e:      # one judge failing shouldn't lose the run; the other still scores it
            verdicts[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
    combined = {}
    for key in ("accuracy", "completeness", "unsupported_claims", "honesty"):
        values = [v[key] for v in verdicts.values() if v.get(key) is not None]
        combined[key] = round(sum(values) / len(values), 2) if values else None
    accs = [v["accuracy"] for v in verdicts.values() if v.get("accuracy") is not None]
    combined["judges_agree"] = (max(accs) - min(accs) < _DISAGREE_POINTS) if len(accs) == 2 else None
    combined["judges"] = verdicts
    combined["reason"] = " | ".join(f"{n}: {v.get('reason', v.get('error', ''))}" for n, v in verdicts.items())[:600]
    return combined


async def rejudge_run(config: Config, store: "BenchStore", run: dict, trap: bool) -> dict:
    """Scores a finished run again from its saved report - no new generation, so a judge fix doesn't cost the
    systems' tokens again. Cited sources are matched to the topic's saved search results by URL."""
    evidence = await store.evidence(run["topic_id"])
    by_url = {(s.get("url") or "").rstrip("/"): s for s in evidence if s.get("content")}
    ids = cited_ids(run.get("report") or "")
    cited = [{**s, **by_url.get((s.get("url") or "").rstrip("/"), {})} for s in run.get("sources") or []
             if s.get("id") in ids]
    run["judge"] = await judge_report(config, run["topic"], run["report"], evidence, trap, cited)
    await store.save_run(run)
    return run


# ── one benchmark run ────────────────────────────────────────────────────────

def new_run(label: str, topic_id: str, topic: str, system: str) -> dict:
    return {"id": str(uuid.uuid4()), "label": label, "topic_id": topic_id, "topic": topic, "system": system,
            "status": "running", "started_at": datetime.utcnow()}


async def execute_run(config: Config, redis: aioredis.Redis, store: BenchStore, run: dict, trap: bool = False) -> dict:
    """Runs one (topic, system), then scores it. Generation usage is metered separately from the judge's."""
    search = snapshot_search(store, run["topic_id"])
    try:
        counters = metering.start()
        started = time.monotonic()
        if run["system"] == "single":
            report, sources = await run_single(config, run["topic"])
        elif run["system"] == "rag":
            report, sources = await run_rag(config, redis, run["topic"], search)
        elif run["system"] == "multi":
            report, sources = await run_multi(config, redis, run["topic"], search)
        else:
            raise ValueError(f"unknown system {run['system']!r}")
        latency = round(time.monotonic() - started, 1)
        usage = dict(counters)
        served = sum(usage["models"].values())
        # Free-tier rate limits make TensorZero fall back to gpt-oss-20b; report how much of the run the main model did
        usage["main_model_share"] = round(usage["models"].get(MAIN_MODEL, 0) / served, 3) if served else None

        metering.start()                      # don't count the judge as the system's cost
        cited = [s for s in sources if s.get("id") in cited_ids(report)]
        judge = await judge_report(config, run["topic"], report, await store.evidence(run["topic_id"]), trap, cited)
        run.update(status="done", report=report, sources=[{k: s.get(k) for k in ("id", "title", "url", "kind")}
                                                         for s in sources],
                   metrics={**report_metrics(report, sources), **usage, "latency_s": latency}, judge=judge)
    except Exception as e:   # recorded on the run; the runner decides whether to retry
        run.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
    run["finished_at"] = datetime.utcnow()
    await store.save_run(run)
    return run


# ── Step 3: critic setups on labelled claims (error injection) ──────────────

CRITIC_MODES = ("single", "panel", "debate")


def evidence_sources(evidence: list[dict], limit: int = 8) -> list[dict]:
    """The topic's snapshotted sources, deduplicated and numbered, as test material for the critics."""
    seen, out = set(), []
    for s in evidence:
        url = (s.get("url") or "").rstrip("/")
        if url in seen or len(s.get("content", "")) < 200:
            continue
        seen.add(url)
        out.append({**s, "id": len(out) + 1})
        if len(out) >= limit:
            break
    return out


async def execute_critic_eval(config: Config, store: BenchStore, run: dict, mode: str, learning=None) -> dict:
    """Scores one critic setup on claims built from a topic's saved sources (same claims for every setup).
    With a LearningService (debate mode), the labels also train critic trust and the debate policy (Step 4a/4b)."""
    from app.critic_eval import evaluate_critic
    try:
        sources = evidence_sources(await store.evidence(run["topic_id"]))
        if not sources:
            raise ValueError("no saved sources for this topic yet - run the benchmark on it first")
        started = time.monotonic()
        learn = learning is not None and mode == "debate"
        result = await evaluate_critic(config, sources, mode, trust=learning.trust if learn else None,
                                       policy=learning.policy if learn else None, learn=learn)
        if learn:
            await learning.save()
        details = result.pop("claims_detail")
        run.update(status="done", report=json.dumps(details), metrics={**result, "latency_s": round(time.monotonic() - started, 1)})
    except Exception as e:
        run.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
    run["finished_at"] = datetime.utcnow()
    await store.save_run(run)
    return run
