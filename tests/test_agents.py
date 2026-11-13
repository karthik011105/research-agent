import asyncio
import json
import re
import pytest
from langgraph.graph import END
from app import agents
from app.agents import (
    OrchestratorAgent, build_graph, cited_ids, clean_kinds, clean_queries, initial_state, normalize_citations,
    parse_json_object, parse_plan, plan_calls, sources_section, response_text, strip_citations,
)
from app.tools.search import SearchUnavailable
from tests.conftest import make_source


def test_parse_json_object_tolerates_surrounding_text():
    assert parse_json_object('Sure!\n```json\n{"queries": ["a"]}\n```') == {"queries": ["a"]}
    assert parse_json_object("no json here") == {}
    assert parse_json_object("{broken") == {}


def test_clean_queries_dedupes_and_limits():
    assert clean_queries(["A", "a ", "", 5, "b", "c"], 2) == ["A", "b"]
    assert clean_queries("not a list", 3) == []


def test_normalize_citations_handles_model_styles():
    text = "EU rules【1】 and US bills【2, 6】 ［3］ news【4†L10-L20】 [5†source] keep [7] and [a link](https://x.y)"
    assert normalize_citations(text) == "EU rules[1] and US bills[2, 6] [3] news[4] [5] keep [7] and [a link](https://x.y)"


async def test_writer_prompt_keeps_unrelated_memory_out_of_the_report(config, monkeypatch):
    prompts = []

    async def fake_llm(config_, function_name, message):
        prompts.append(message)
        return "report"

    monkeypatch.setattr(agents, "_tz_call", fake_llm)
    await agents.WriterAgent(config).run("AI regulation", "### Q1\n- f [1]", True, "AI chips report [9]", "")
    assert "Never mention it" in prompts[0] and "AI chips report" in prompts[0] and "[9]" not in prompts[0]
    assert "Cover every sub-question" in prompts[0]


async def test_writer_output_citations_feed_the_sources_list(config, monkeypatch):
    async def fake_llm(config_, function_name, message):
        return "Timetable published【1】; Senate bill【6】."

    monkeypatch.setattr(agents, "_tz_call", fake_llm)
    report = await agents.WriterAgent(config).run("topic", "- f [1]", True, "", "")
    sources = [{**make_source(i), "id": i} for i in range(1, 7)]
    assert cited_ids(report) == {1, 6}
    assert "[6] Page 6" in sources_section(report, sources)


def test_cited_ids_reads_single_and_grouped_citations():
    assert cited_ids("x [1] y [2, 4] z [10]") == {1, 2, 4, 10}


def test_sources_section_lists_only_cited_sources():
    sources = [{**make_source(i), "id": i} for i in (1, 2, 3)]
    section = sources_section("Claim [3]. Other [1].", sources)
    assert "[1] Page 1" in section and "[3] Page 3" in section and "[2]" not in section
    assert sources_section("No citations.", sources) == ""


def test_response_text_skips_reasoning_blocks():
    body = {"content": [{"type": "thought", "text": "let me think"}, {"type": "text", "text": "Answer"}]}
    assert response_text(body) == "Answer"
    with pytest.raises(ValueError):
        response_text({"content": [{"type": "thought", "text": "only thinking"}]})


def test_strip_citations_removes_markers_and_sources_list():
    old = "## Summary\nChips grew 20% [1] and [2, 3].\n\n## Sources\n[1] Old page — https://old.com"
    assert strip_citations(old) == "## Summary\nChips grew 20% and."


def test_parse_plan_validates_sub_questions():
    reply = json.dumps({"sub_questions": [
        {"question": "What methods exist?", "queries": ["llm guardrail methods", "llm guardrail methods "],
         "kinds": ["paper", "PAPER", "tv"]},
        {"question": "Who ships them?", "queries": [], "kinds": "news"},
        {"queries": ["item without a question"]},
        "not a dict",
        {"question": "Fourth", "queries": ["x"]},
    ]})
    plan = parse_plan(reply, "topic", max_subquestions=3, max_queries=2)
    assert [sq["id"] for sq in plan] == [1, 2, 3]
    assert plan[0] == {"id": 1, "question": "What methods exist?", "queries": ["llm guardrail methods"], "kinds": ["paper"]}
    assert plan[1]["queries"] == ["Who ships them?"] and plan[1]["kinds"] == ["news"]
    assert plan[2]["question"] == "item without a question" and plan[2]["kinds"] == ["web"]


def test_parse_plan_falls_back_to_legacy_shape_and_topic():
    legacy = parse_plan('{"queries": ["a", "b"]}', "topic", 3, 2)
    assert legacy == [{"id": 1, "question": "topic", "queries": ["a", "b"], "kinds": ["web"]}]
    assert parse_plan("I cannot help", "AI chips", 3, 2)[0]["queries"] == ["AI chips"]


def test_clean_kinds_defaults_to_web_and_caps_at_two():
    assert clean_kinds(["news", "paper", "community"]) == ["news", "paper"]
    assert clean_kinds(None) == ["web"] and clean_kinds("community") == ["community"]


def test_plan_calls_shares_paid_budget_round_robin():
    subqs = [
        {"id": 1, "question": "a", "queries": ["a1", "a2"], "kinds": ["web"]},
        {"id": 2, "question": "b", "queries": ["b1", "b2"], "kinds": ["news"]},
        {"id": 3, "question": "c", "queries": ["c1", "c2"], "kinds": ["paper", "web"]},
    ]
    calls = plan_calls(subqs, web_limit=4)
    assert calls[1] == [("a1", "web", True), ("a2", "web", True)]
    # b's second query comes after the paid budget is spent, so it only reaches the free news feed
    assert calls[2] == [("b1", "news", True), ("b2", "news", False)]
    # papers are free and always searched; c's second web query is dropped once the budget is gone
    assert calls[3] == [("c1", "paper", False), ("c1", "web", True), ("c2", "paper", False)]


def _state(**overrides):
    state = initial_state("topic", "s", [], "")
    state.update(overrides)
    return state


@pytest.fixture
def orchestrator(config, redis):
    return OrchestratorAgent(config, redis)


def test_route_after_verify_rewrites_until_limit(orchestrator):
    assert orchestrator.route_after_verify(_state(verified=False, iterations=1)) == "write"
    assert orchestrator.route_after_verify(_state(verified=False, iterations=2)) == END
    assert orchestrator.route_after_verify(_state(verified=True, iterations=1)) == END


async def test_gap_check_skipped_without_live_sources_or_at_round_limit(orchestrator, monkeypatch):
    async def fail(*args):
        raise AssertionError("gap check must not call the LLM")

    monkeypatch.setattr(orchestrator.gap_agent, "run", fail)
    assert (await orchestrator.gap_check_node(_state(live_sources=False, search_rounds=1)))["needs_more"] is False
    assert (await orchestrator.gap_check_node(_state(live_sources=True, search_rounds=2)))["needs_more"] is False


class FakeLLM:
    """Scripted TensorZero replies, keyed by function and (for research_plan) planner vs gap check."""

    def __init__(self, critic_replies, gap_reply, plan=None):
        self.critic_replies = list(critic_replies)
        self.gap_reply = gap_reply
        self.plan = plan or {"sub_questions": [
            {"question": "What techniques exist?", "queries": ["q1"], "kinds": ["paper"]},
            {"question": "What do practitioners say?", "queries": ["q2"], "kinds": ["web", "community"]},
        ]}
        self.calls = []
        self.active = self.max_active = 0

    async def __call__(self, config, function_name, message):
        self.calls.append(function_name)
        if function_name == "research_plan":
            if "Queries already searched" in message:
                return json.dumps(self.gap_reply)
            return "Plan:\n" + json.dumps(self.plan)
        if function_name == "research_extract":
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            ids = re.findall(r'<source id="(\d+)"', message)
            return "\n".join(f"- fact from source [{i}]" for i in ids)
        if function_name == "research_summarize":
            return "- fact from memory"
        if function_name == "report_write":
            return "## Executive Summary\nThings happened [1] and more [3]."
        if function_name == "research_verify":
            return self.critic_replies.pop(0)
        raise AssertionError(function_name)


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    async def search_kind(redis_, config_, query, kind, allow_paid=True):
        calls.append((query, kind, allow_paid))
        provider = {"web": "tavily", "news": "tavily", "paper": "arxiv", "community": "hackernews"}[kind]
        return [{**make_source(0, url=f"https://{kind}.example/{query}/{n}"), "provider": provider, "kind": kind}
                for n in range(2)]

    async def no_scrape(redis_, config_, sources):
        return sources

    monkeypatch.setattr(agents, "search_kind", search_kind)
    monkeypatch.setattr(agents, "enrich_sources", no_scrape)
    return calls


async def test_full_graph_fans_out_per_sub_question_with_gap_round(config, redis, monkeypatch, fake_search):
    config.search_max_sources = 12
    llm = FakeLLM(critic_replies=["NO\n- claim X is uncited", "YES"], gap_reply={"queries": ["q3"], "kind": "news"})
    monkeypatch.setattr(agents, "_tz_call", llm)

    final = await build_graph(config, redis).ainvoke(initial_state("LLM guardrails", "s1", [], ""))

    assert sorted(fake_search) == [("q1", "paper", False), ("q2", "community", False), ("q2", "web", True), ("q3", "news", True)]
    assert final["search_rounds"] == 2 and final["live_sources"] is True
    ids = [s["id"] for s in final["sources"]]
    assert ids == list(range(1, 9))                                   # numbered once, globally: no clashes
    assert {s["subq"] for s in final["sources"]} == {1, 2, 3}
    assert {s["kind"] for s in final["sources"]} == {"paper", "web", "community", "news"}
    # one extract per sub-question per round: 2 in the first round, 1 for the gap sub-question
    assert llm.calls.count("research_extract") == 3
    facts = final["facts"]
    assert facts.index("### What techniques exist?") < facts.index("### What do practitioners say?")
    assert "### Additional findings: q3" in facts
    # each branch saw only its own sources
    paper_ids = {s["id"] for s in final["sources"] if s["subq"] == 1}
    paper_section = facts.split("### What do practitioners say?")[0]
    assert set(map(int, re.findall(r"\[(\d+)\]", paper_section))) == paper_ids
    assert final["iterations"] == 2 and final["verified"] is True
    assert "## Sources" in final["report"]
    assert llm.calls.count("research_plan") == 2      # planner + one gap check (round limit reached after)


async def test_graph_limits_parallel_llm_calls(config, redis, monkeypatch, fake_search):
    config.llm_concurrency = 1
    config.search_max_rounds = 1
    llm = FakeLLM(critic_replies=["YES"], gap_reply={"queries": []}, plan={"sub_questions": [
        {"question": f"Q{i}", "queries": [f"q{i}"], "kinds": ["paper"]} for i in range(3)]})
    monkeypatch.setattr(agents, "_tz_call", llm)
    final = await build_graph(config, redis).ainvoke(initial_state("topic", "s1", [], ""))
    assert llm.calls.count("research_extract") == 3 and llm.max_active == 1
    assert final["search_rounds"] == 1


async def test_graph_falls_back_to_model_knowledge_when_search_unavailable(config, redis, monkeypatch):
    llm = FakeLLM(critic_replies=["YES"], gap_reply={"queries": ["never used"]})

    async def no_search(*args, **kwargs):
        raise SearchUnavailable("No search API key configured")

    monkeypatch.setattr(agents, "_tz_call", llm)
    monkeypatch.setattr(agents, "search_kind", no_search)

    final = await build_graph(config, redis).ainvoke(initial_state("AI chips", "s1", [], ""))

    assert final["live_sources"] is False
    assert final["sources"] == []
    assert "## Sources" not in final["report"]
    assert "research_summarize" in llm.calls and "research_extract" not in llm.calls
    assert llm.calls.count("research_plan") == 1   # no gap check without live sources
    assert final["error"] == "search: No search API key configured"
