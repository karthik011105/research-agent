import asyncio
import json
import pytest
from app import agents, bench, metering
from app.bench import execute_run, judge_report, new_run, report_metrics, snapshot_search
from tests.conftest import make_source


class MemoryStore:
    def __init__(self):
        self.snapshots, self.runs = {}, {}

    async def get_snapshot(self, topic_id, query, kind):
        return self.snapshots.get((topic_id, query, kind))

    async def put_snapshot(self, topic_id, query, kind, results):
        self.snapshots.setdefault((topic_id, query, kind), results)

    async def evidence(self, topic_id):
        return [s for (t, _, _), rs in self.snapshots.items() if t == topic_id for s in rs]

    async def save_run(self, run):
        self.runs[run["id"]] = dict(run)


def test_report_metrics_measures_citations_by_code():
    sources = [{**make_source(i), "id": i} for i in (1, 2)]
    report = ("## Executive Summary\nSales grew 20% in 2026 [1]. Costs fell 5% last year. "
              "A claim citing a missing source [7].\n## Key Findings\nMore detail here without numbers.\n"
              "## Analysis\nText.\n## Conclusion\nDone.\n\n## Sources\n[1] Page 1 - https://example.com/1")
    m = report_metrics(report, sources)
    assert m["citations"] == 2 and m["citation_validity"] == 0.5        # [7] doesn't exist
    assert m["numeric_claims"] == 2 and m["numeric_cited_ratio"] == 0.5  # "Costs fell 5%" is uncited
    assert m["sections_present"] == 4 and m["sources"] == 2
    assert report_metrics("No sources, no numbers here at all.", [])["citation_validity"] is None


async def test_snapshot_search_records_once_then_replays(config, redis, monkeypatch):
    calls = []

    async def live_search(redis_, config_, query, kind, allow_paid=True):
        calls.append((query, kind))
        return [make_source(len(calls))]

    monkeypatch.setattr(bench, "search_kind", live_search)
    store = MemoryStore()
    search = snapshot_search(store, "t1")
    first = await search(redis, config, "AI chips 2026", "web")
    again = await search(redis, config, "ai  chips 2026", "web")         # normalised query: same snapshot
    other_kind = await search(redis, config, "AI chips 2026", "news")
    assert first == again and calls == [("AI chips 2026", "web"), ("AI chips 2026", "news")]
    assert other_kind != first
    assert len(await store.evidence("t1")) == 2 and await store.evidence("other") == []


async def test_metering_sums_calls_across_parallel_tasks():
    counters = metering.start()

    async def call():
        await asyncio.sleep(0)
        metering.record(llm_calls=1, input_tokens=100)

    await asyncio.gather(*(asyncio.create_task(call()) for _ in range(3)))
    assert counters["llm_calls"] == 3 and counters["input_tokens"] == 300


async def test_two_judges_are_averaged_and_disagreement_flagged(config, monkeypatch):
    prompts = []
    replies = {
        "research_judge": '{"accuracy": 14, "completeness": "7", "unsupported_claims": 2, "honesty": 3, "reason": "a"}',
        "research_judge_alt": 'ok {"accuracy": 4, "completeness": 5, "unsupported_claims": 4, "honesty": 5, "reason": "b"}',
    }

    async def fake_llm(config_, function_name, message):
        prompts.append((function_name, message))
        return replies[function_name]

    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    j = await judge_report(config, "Nobel Prize in Mathematics 2026", "report", [make_source(1)], trap=True)
    assert [f for f, _ in prompts] == ["research_judge", "research_judge_alt"]
    assert prompts[0][1] == prompts[1][1] and "TRAP topic" in prompts[0][1]       # identical prompt for both
    assert j["judges"]["qwen"]["accuracy"] == 10 and j["judges"]["gpt_oss"]["accuracy"] == 4   # clamped
    assert (j["accuracy"], j["completeness"], j["unsupported_claims"], j["honesty"]) == (7, 6, 3, 4)
    assert j["judges_agree"] is False                                              # 10 vs 4: flagged


async def test_one_judge_failing_still_scores(config, monkeypatch):
    async def fake_llm(config_, function_name, message):
        if function_name == "research_judge":
            raise RuntimeError("LLM daily limit reached")
        return '{"accuracy": 8, "completeness": 7, "unsupported_claims": 1}'

    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    j = await judge_report(config, "topic", "report", [], trap=False)
    assert j["accuracy"] == 8 and j["judges_agree"] is None and "error" in j["judges"]["qwen"]
    assert j["honesty"] is None


def test_evidence_puts_cited_sources_first_with_more_text():
    long_text = "Model Y sold 408,158 units. " + "x" * 3000
    cited = [{**make_source(7, url="https://ev.example/sales", content=long_text), "id": 7, "title": "EV sales 2026"}]
    pool = [make_source(i, content=f"Pool fact {i}. " * 100) for i in range(30)] + cited
    block = bench.evidence_block(pool, cited)
    assert block.startswith("- EV sales 2026") and "408,158" in block
    first_entry = block.split("\n- ")[0]
    assert len(first_entry) > 1400                       # cited source: ~1500 chars, not 450
    assert block.count("EV sales 2026") == 1             # listed once, though it is also in the pool
    assert len(block) <= bench._EVIDENCE_CHARS


async def test_execute_run_scores_each_system_and_meters_generation_only(config, redis, monkeypatch):
    async def fake_llm(config_, function_name, message):
        metering.record(llm_calls=1, input_tokens=10, output_tokens=5)
        if function_name.startswith("research_judge"):
            return json.dumps({"accuracy": 8, "completeness": 7, "unsupported_claims": 1, "reason": "fine"})
        return "## Executive Summary\nGrowth was 12% [1].\n## Conclusion\nGood."

    async def live_search(redis_, config_, query, kind, allow_paid=True):
        metering.record(paid_searches=1)
        return [make_source(1), make_source(2)]

    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    monkeypatch.setattr(bench, "search_kind", live_search)
    store = MemoryStore()

    single = await execute_run(config, redis, store, new_run("v1", "t1", "AI chips", "single"))
    rag = await execute_run(config, redis, store, new_run("v1", "t1", "AI chips", "rag"))
    assert single["status"] == rag["status"] == "done"
    assert single["metrics"]["llm_calls"] == 1 and single["metrics"]["paid_searches"] == 0   # judge not counted
    assert rag["metrics"]["paid_searches"] == 1 and rag["metrics"]["citation_validity"] == 1.0
    assert "## Sources" in rag["report"] and single["judge"]["accuracy"] == 8
    # second rag run replays the snapshot: no new paid search
    rag2 = await execute_run(config, redis, store, new_run("v1", "t1", "AI chips", "rag"))
    assert rag2["metrics"]["paid_searches"] == 0
    assert all(r["status"] == "done" for r in store.runs.values())


async def test_execute_run_records_errors_instead_of_raising(config, redis, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("LLM daily limit reached for all models")

    monkeypatch.setattr(bench, "_tz_call", broken)
    store = MemoryStore()
    run = await execute_run(config, redis, store, new_run("v1", "t1", "AI chips", "single"))
    assert run["status"] == "error" and "daily limit" in run["error"]
    assert store.runs[run["id"]]["finished_at"] is not None


async def test_multi_system_uses_the_snapshot_search(config, redis, monkeypatch):
    seen = []

    async def live_search(redis_, config_, query, kind, allow_paid=True):
        seen.append(query)
        return [{**make_source(len(seen), url=f"https://x.example/{len(seen)}"), "kind": kind}]

    async def fake_llm(config_, function_name, message):
        if function_name == "research_plan" and "Queries already searched" not in message:
            return json.dumps({"sub_questions": [{"question": "Q", "queries": ["q1"], "kinds": ["web"]}]})
        if function_name == "research_plan":
            return '{"queries": []}'
        if function_name == "research_verify":
            return "YES"
        if function_name.startswith("research_judge"):
            return '{"accuracy": 9, "completeness": 9, "unsupported_claims": 0}'
        return "## Executive Summary\nFact [1]."

    async def no_scrape(redis_, config_, sources):
        return sources

    monkeypatch.setattr(bench, "search_kind", live_search)
    monkeypatch.setattr(agents, "_tz_call", fake_llm)
    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    monkeypatch.setattr(agents, "enrich_sources", no_scrape)
    store = MemoryStore()
    run = await execute_run(config, redis, store, new_run("v1", "t9", "topic", "multi"))
    assert run["status"] == "done", run.get("error")
    assert seen == ["q1"] and ("t9", "q1", "web") in store.snapshots


def test_served_models_reads_tensorzero_raw_responses():
    body = {"raw_response": [
        {"provider_type": "groq", "data": json.dumps({"model": "openai/gpt-oss-120b", "error": "rate limited"})},
        {"provider_type": "groq", "data": json.dumps({"model": "openai/gpt-oss-20b", "choices": []})},
        {"provider_type": "groq", "data": "not json"},
    ]}
    assert agents.served_models(body) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert agents.served_models({}) == []


async def test_main_model_share_reflects_fallbacks(config, redis, monkeypatch):
    async def fake_llm(config_, function_name, message):
        if not function_name.startswith("research_judge"):
            metering.record(llm_calls=1)
            metering.record_model("openai/gpt-oss-20b")      # writer fell back
        if function_name.startswith("research_judge"):
            return '{"accuracy": 5, "completeness": 5, "unsupported_claims": 0}'
        return "## Executive Summary\nx"

    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    run = await execute_run(config, redis, MemoryStore(), new_run("v1", "t1", "topic", "single"))
    assert run["metrics"]["main_model_share"] == 0.0 and run["metrics"]["models"] == {"openai/gpt-oss-20b": 1}


def test_evidence_sources_dedupes_numbers_and_skips_thin_pages():
    ev = [make_source(1), make_source(1, url="https://example.com/1/"), make_source(2, content="short"), make_source(3)]
    out = bench.evidence_sources(ev)
    assert [s["id"] for s in out] == [1, 2] and [s["url"] for s in out] == ["https://example.com/1", "https://example.com/3"]


async def test_critic_eval_run_records_scores(config, monkeypatch):
    async def fake_eval(config_, sources, mode, n=16, seed=7, **kwargs):
        return {"caught": 0.75, "false_alarms": 0.1, "llm_calls": 4, "claims_detail": [{"text": "x", "flagged": True}]}

    monkeypatch.setattr("app.critic_eval.evaluate_critic", fake_eval)
    store = MemoryStore()
    store.snapshots[("t1", "q", "web")] = [make_source(1), make_source(2)]
    run = await bench.execute_critic_eval(config, store, new_run("c1", "t1", "critic eval", "critic_debate"), "debate")
    assert run["status"] == "done" and run["metrics"]["caught"] == 0.75 and json.loads(run["report"])[0]["flagged"]
    empty = await bench.execute_critic_eval(config, MemoryStore(), new_run("c1", "t9", "x", "critic_panel"), "panel")
    assert empty["status"] == "error" and "no saved sources" in empty["error"]


def test_every_cited_source_reaches_the_judge_even_when_a_report_cites_twelve():
    # v2 run: with 1,500 characters per source only ~6 cited sources fit, and facts from the rest looked invented
    cited = [{**make_source(i, url=f"https://src.example/{i}", content=f"FACT-{i} " + "y" * 3000), "id": i,
              "title": f"Source {i}"} for i in range(1, 13)]
    pool = [make_source(i, content="pool " * 400) for i in range(40)]
    block = bench.evidence_block(pool, cited)
    assert all(f"FACT-{i} " in block for i in range(1, 13))
    assert len(block) <= bench._EVIDENCE_CHARS
    # a report with few sources still gets their full text
    few = bench.evidence_block(pool, cited[:3])
    assert len(few.split("\n- ")[0]) > 1400


async def test_rejudge_scores_a_saved_report_again_without_generating(monkeypatch):
    store = MemoryStore()
    await store.put_snapshot("t1", "q", "web", [make_source(1, url="https://a.example/1", content="Sales rose 12%. " * 20)])
    prompts = []

    async def fake_llm(config_, function_name, message):
        assert function_name.startswith("research_judge"), function_name   # judging only, no report writing
        prompts.append(message)
        return json.dumps({"accuracy": 9, "completeness": 8, "unsupported_claims": 0, "reason": "ok"})

    monkeypatch.setattr(bench, "_tz_call", fake_llm)
    run = {**new_run("v2", "t1", "EV sales", "multi"), "status": "done", "report": "Sales rose 12% [1].",
           "sources": [{"id": 1, "title": "Page 1", "url": "https://a.example/1", "kind": "web"}],
           "judge": {"accuracy": 4}}
    out = await bench.rejudge_run(None, store, run, trap=False)
    assert out["judge"]["accuracy"] == 9 and store.runs[run["id"]]["judge"]["accuracy"] == 9
    assert all("Sales rose 12%" in p for p in prompts)           # the cited source's text came from the snapshot


def test_single_call_baseline_may_write_from_knowledge():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "tensorzero" / "templates"
    for name in ("report_write_system", "report_write_cite_numbers_system"):
        text = (root / f"{name}.minijinja").read_text(encoding="utf-8")
        assert "use your own knowledge, write the report from what you know" in text and "Do not refuse." in text
