import json
import pytest
from langgraph.graph import END
from app import agents
from app.agents import (
    OrchestratorAgent, build_graph, cited_ids, clean_queries, initial_state, parse_json_object, sources_section,
    response_text, strip_citations,
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

    def __init__(self, critic_replies, gap_queries):
        self.critic_replies = list(critic_replies)
        self.gap_queries = gap_queries
        self.calls = []

    async def __call__(self, config, function_name, message):
        self.calls.append(function_name)
        if function_name == "research_plan":
            if "Queries already searched" in message:
                return json.dumps({"queries": self.gap_queries})
            return 'Here you go: {"search_topic": "news", "queries": ["q1", "q2"]}'
        if function_name in ("research_extract", "research_summarize"):
            return "- fact [1]\n- fact [2]"
        if function_name == "report_write":
            return "## Executive Summary\nThings happened [1] and more [2]."
        if function_name == "research_verify":
            return self.critic_replies.pop(0)
        raise AssertionError(function_name)


async def test_full_graph_with_gap_round_and_critic_rewrite(config, redis, monkeypatch):
    llm = FakeLLM(critic_replies=["NO\n- claim X is uncited", "YES"], gap_queries=["q3"])
    searches = []

    async def fake_search_many(redis_, config_, queries, existing, topic):
        searches.append((list(queries), topic))
        start = len(existing) + 1
        return existing + [{**make_source(i), "id": i} for i in range(start, start + 2)]

    monkeypatch.setattr(agents, "_tz_call", llm)
    monkeypatch.setattr(agents, "search_many", fake_search_many)

    final = await build_graph(config, redis).ainvoke(initial_state("AI chips", "s1", [], ""))

    assert searches == [(["q1", "q2"], "news"), (["q3"], "news")]
    assert final["search_rounds"] == 2
    assert final["live_sources"] is True
    assert len(final["sources"]) == 4
    assert final["iterations"] == 2          # rewritten once after critic NO
    assert final["verified"] is True
    assert "## Sources" in final["report"] and "https://example.com/1" in final["report"]
    assert llm.calls.count("research_plan") == 2  # planner + one gap check (round limit reached after)


async def test_graph_falls_back_to_model_knowledge_when_search_unavailable(config, redis, monkeypatch):
    llm = FakeLLM(critic_replies=["YES"], gap_queries=["never used"])

    async def no_search(*args):
        raise SearchUnavailable("No search API key configured")

    monkeypatch.setattr(agents, "_tz_call", llm)
    monkeypatch.setattr(agents, "search_many", no_search)

    final = await build_graph(config, redis).ainvoke(initial_state("AI chips", "s1", [], ""))

    assert final["live_sources"] is False
    assert final["sources"] == []
    assert "## Sources" not in final["report"]
    assert "research_summarize" in llm.calls and "research_extract" not in llm.calls
    assert llm.calls.count("research_plan") == 1   # no gap check without live sources
    assert final["error"].startswith("search:")
