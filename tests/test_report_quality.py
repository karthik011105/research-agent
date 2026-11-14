"""Fixes from reviewing a live report ("what are new ai new today"): recent-only search for time-sensitive
topics, a source reserve for the gap round, unresolved fact-check points shown with the report, and the
writer / critic rules against merged figures and unsupported recommendations."""
import json
import pathlib
from datetime import date, timedelta
import httpx
import pytest
import respx
from app import agents
from app.agents import (
    OrchestratorAgent, REVIEW_NOTES_HEADING, build_graph, initial_state, parse_recency, review_notes,
    with_review_notes,
)
from app.monitoring import what_changed
from app.tools import providers, search
from app.tools.search import _cache_key, is_stale, merge_sources, prefer_recent, search_kind, web_search
from tests.conftest import make_source

TODAY = date(2026, 11, 13)
TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "tensorzero" / "templates"


def dated(n, days_old, kind="web"):
    return {**make_source(n, kind=kind), "published_date": (TODAY - timedelta(days=days_old)).isoformat()}


# ── which sources count as too old ──

def test_stale_means_clearly_older_than_the_topic_asks_for():
    assert not is_stale(dated(1, 10), "week", TODAY)            # within twice the window (pages get updated)
    assert is_stale(dated(1, 20), "week", TODAY)
    assert not is_stale(dated(1, 50), "month", TODAY) and is_stale(dated(1, 70), "month", TODAY)
    assert not is_stale(dated(1, 400), "", TODAY)               # topic without a date limit
    assert not is_stale(make_source(1), "week", TODAY)          # undated pages are kept
    assert not is_stale({**make_source(1), "published_date": "2 days ago"}, "week", TODAY)


def test_papers_get_at_least_a_year_even_for_this_weeks_news():
    assert not is_stale(dated(1, 200, kind="paper"), "week", TODAY)
    assert is_stale(dated(1, 800, kind="paper"), "week", TODAY)


def test_prefer_recent_drops_old_pages_but_never_leaves_a_report_without_sources():
    fresh, old = [dated(1, 2), dated(2, 3), dated(3, 1)], [dated(4, 200), dated(5, 240)]
    kept, dropped = prefer_recent(old[:1] + fresh + old[1:], "week", TODAY)
    assert [s["url"] for s in kept] == [s["url"] for s in fresh] and dropped == 2
    # a thin day: too few recent pages, so the old ones stay - after the recent one
    kept, dropped = prefer_recent([old[0], fresh[0], old[1]], "week", TODAY)
    assert [s["url"] for s in kept] == [fresh[0]["url"], old[0]["url"], old[1]["url"]] and dropped == 0
    assert prefer_recent(old, "", TODAY) == (old, 0)


# ── searching with a date limit ──

def test_date_limited_search_has_its_own_cache_entry():
    assert _cache_key("ai news", "news") == _cache_key("ai news", "news", "")
    assert _cache_key("ai news", "news", "week") != _cache_key("ai news", "news")


@respx.mock
async def test_tavily_and_brave_are_asked_for_recent_results(config, redis):
    tavily = respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    await web_search(redis, config, "ai launches", "news", recency="week")
    assert json.loads(tavily.calls[0].request.content)["time_range"] == "week"
    await web_search(redis, config, "ai launches", "news")
    assert "time_range" not in json.loads(tavily.calls[1].request.content)   # a separate, undated search

    config.brave_api_key = "brave-test"
    respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(500))
    brave = respx.get(search._BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": []}}))
    await web_search(redis, config, "chips", recency="month")
    assert brave.calls[0].request.url.params["freshness"] == "pm"


@pytest.fixture
def no_throttle(monkeypatch):
    async def instant(provider):
        return None
    monkeypatch.setattr(providers, "throttle", instant)


@respx.mock
async def test_free_providers_get_date_filters(config, redis, no_throttle):
    openalex = respx.get(providers.OPENALEX_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(providers.ARXIV_URL).mock(return_value=httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'/>"))
    respx.get(providers.SEMANTIC_SCHOLAR_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    await search_kind(redis, config, "llm benchmarks", "paper", recency="month")
    assert "from_publication_date:" in openalex.calls[0].request.url.params["filter"]

    hn = respx.get(providers.HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": []}))
    await search_kind(redis, config, "ai tools", "community", recency="week")
    assert hn.calls[0].request.url.params["numericFilters"].startswith("created_at_i>")

    gdelt = respx.get(providers.GDELT_URL).mock(return_value=httpx.Response(200, json={"articles": []}))
    config.enabled_providers = ["gdelt"]
    await search_kind(redis, config, "ai launches", "news", recency="week")
    assert gdelt.calls[0].request.url.params["timespan"] == "1w"


# ── planner: recency ──

def test_recency_comes_from_the_planner_or_from_words_like_today():
    assert parse_recency('{"recency": "month", "sub_questions": []}', "quantum computing") == "month"
    assert parse_recency('{"recency": "", "sub_questions": []}', "latest chips") == ""   # the planner decided
    assert parse_recency('{"sub_questions": []}', "what are new ai new today") == "week"
    assert parse_recency('{"sub_questions": []}', "latest small language models") == "month"
    assert parse_recency('{"sub_questions": []}', "history of the transistor") == ""
    assert parse_recency('{"recency": "forever"}', "x") == ""


class ScriptedLLM:
    def __init__(self, plan, critic_replies, gap_reply=None):
        self.plan, self.critic_replies, self.gap_reply = plan, list(critic_replies), gap_reply or {"queries": []}
        self.calls, self.prompts = [], {}

    async def __call__(self, config, function_name, message, **kwargs):
        self.calls.append(function_name)
        self.prompts.setdefault(function_name, []).append(message)
        if function_name == "research_plan":
            return json.dumps(self.gap_reply if "Queries already searched" in message else self.plan)
        if function_name == "research_extract":
            return "- a fact [1]"
        if function_name == "report_write":
            return "## Executive Summary\nSomething launched [1]."
        if function_name == "research_verify":
            return self.critic_replies.pop(0)
        raise AssertionError(function_name)


@pytest.fixture
def no_scrape(monkeypatch):
    async def passthrough(redis_, config_, sources):
        return sources
    monkeypatch.setattr(agents, "enrich_sources", passthrough)


async def test_time_sensitive_topic_searches_recent_only_and_skips_old_pages(config, redis, monkeypatch, no_scrape):
    seen = []
    today = date.today()

    async def search_kind_(redis_, config_, query, kind, allow_paid=True, recency=""):
        seen.append(recency)
        return [{**make_source(n, url=f"https://{kind}.example/{n}", kind=kind),
                 "published_date": (today - timedelta(days=age)).isoformat()}
                for n, age in enumerate([1, 2, 3, 300])]

    llm = ScriptedLLM({"recency": "week", "sub_questions": [
        {"question": "What launched?", "queries": ["ai launches"], "kinds": ["news"]}]}, ["YES"])
    monkeypatch.setattr(agents, "search_kind", search_kind_)
    monkeypatch.setattr(agents, "_tz_call", llm)
    config.search_max_rounds = 1
    final = await build_graph(config, redis).ainvoke(initial_state("what are new ai new today", "s", [], ""))

    assert final["recency"] == "week" and seen == ["week"]
    assert [s["url"] for s in final["sources"]] == [f"https://news.example/{n}" for n in range(3)]
    assert "lead with the newest findings" in llm.prompts["report_write"][0]


async def test_injected_searches_get_no_date_limit(config, redis, monkeypatch, no_scrape):
    async def replay(redis_, config_, query, kind, allow_paid=True):    # benchmark snapshots take no recency
        return [make_source(1, kind=kind)]

    llm = ScriptedLLM({"recency": "week", "sub_questions": [
        {"question": "Q", "queries": ["q"], "kinds": ["web"]}]}, ["YES"])
    monkeypatch.setattr(agents, "_tz_call", llm)
    config.search_max_rounds = 1
    final = await build_graph(config, redis, search_fn=replay).ainvoke(initial_state("latest ai", "s", [], ""))
    assert final["live_sources"] is True


# ── source reserve for the gap round ──

async def test_first_round_leaves_room_for_the_gap_round(config, redis, monkeypatch, no_scrape):
    async def search_kind_(redis_, config_, query, kind, allow_paid=True, recency=""):
        return [make_source(n, url=f"https://{query}.example/{n}", kind=kind) for n in range(10)]

    llm = ScriptedLLM({"sub_questions": [{"question": "Q", "queries": ["first"], "kinds": ["web"]}]},
                      ["YES"], gap_reply={"queries": ["second"], "kind": "web"})
    monkeypatch.setattr(agents, "search_kind", search_kind_)
    monkeypatch.setattr(agents, "_tz_call", llm)
    config.search_max_sources = 8
    final = await build_graph(config, redis).ainvoke(initial_state("topic", "s", [], ""))
    urls = [s["url"] for s in final["sources"]]
    assert len(urls) == 8
    assert sum("first" in u for u in urls) == 5 and sum("second" in u for u in urls) == 3
    gap_prompt = [p for p in llm.prompts["research_plan"] if "Queries already searched" in p][0]
    assert "do not widen it to related subjects" in gap_prompt


async def test_gap_check_is_skipped_when_every_source_slot_is_taken(config, redis, monkeypatch):
    orchestrator = OrchestratorAgent(config, redis)

    async def fail(*args):
        raise AssertionError("no LLM call when there is no room for new sources")

    monkeypatch.setattr(orchestrator.gap_agent, "run", fail)
    state = initial_state("topic", "s", [], "")
    state.update(live_sources=True, search_rounds=1,
                 sources=[{**make_source(i), "id": i} for i in range(1, config.search_max_sources + 1)])
    assert (await orchestrator.gap_check_node(state))["needs_more"] is False


def test_merge_sources_takes_an_explicit_limit(config):
    merged = merge_sources([[make_source(i) for i in range(10)]], [], config, limit=3)
    assert [s["id"] for s in merged] == [1, 2, 3]


# ── unresolved fact-check points ──

def test_review_notes_list_what_the_single_critic_still_flagged():
    state = {"verified": False, "claim_verdicts": [],
             "critic_feedback": "NO\n- \"94% on GPQA\" mixes Gemini's and Claude's scores\n* Analysis cites nothing"}
    assert review_notes(state) == ["\"94% on GPQA\" mixes Gemini's and Claude's scores", "Analysis cites nothing"]
    assert review_notes({"verified": False, "critic_feedback": "NO: the summary misattributes a figure"}) == [
        "the summary misattributes a figure"]
    assert review_notes({"verified": False, "critic_feedback": "NO"}) == [
        "The fact-checker did not approve the final draft."]
    assert review_notes({"verified": True, "critic_feedback": ""}) == []


def test_review_notes_use_the_panels_claim_verdicts():
    state = {"verified": False, "critic_feedback": "x", "claim_verdicts": [
        {"text": "A rose 5% [1]", "decision": "keep"},
        {"text": "B is open-source [2]", "decision": "fix", "fix": "source does not say open-source"},
        {"text": "C won [3]", "decision": "remove", "fix": ""},
    ]}
    assert review_notes(state) == ["B is open-source [2] (source does not say open-source)", "C won [3]"]


async def test_final_report_carries_the_notes_when_the_last_draft_is_rejected(config, redis, monkeypatch, no_scrape):
    async def search_kind_(redis_, config_, query, kind, allow_paid=True, recency=""):
        return [make_source(1, kind=kind)]

    llm = ScriptedLLM({"sub_questions": [{"question": "Q", "queries": ["q"], "kinds": ["web"]}]},
                      ["NO\n- claim A is wrong", "NO\n- claim B is still uncited"])
    monkeypatch.setattr(agents, "search_kind", search_kind_)
    monkeypatch.setattr(agents, "_tz_call", llm)
    config.search_max_rounds = 1
    final = await build_graph(config, redis).ainvoke(initial_state("topic", "s", [], ""))
    assert final["verified"] is False and final["iterations"] == 2
    notes = review_notes(final)
    assert notes == ["claim B is still uncited"]          # from the last check, not the first
    report = with_review_notes(final["report"], notes)
    assert report.index("## Sources") < report.index(REVIEW_NOTES_HEADING)
    assert report.endswith("- claim B is still uncited")
    assert with_review_notes("r", []) == "r"


def test_monitor_diffs_ignore_the_fact_check_notes():
    old = "## Key Findings\n- Chip sales rose 5% in Q3 [1]."
    new = with_review_notes(old, ["Chip sales of a brand new company doubled overnight [2]"])
    assert what_changed(old, new)["new"] == []


# ── prompts ──

def test_writer_and_critic_prompts_forbid_merged_figures_and_unsupported_advice():
    for name in ("report_write_system", "report_write_cite_numbers_system"):
        text = (TEMPLATES / f"{name}.minijinja").read_text(encoding="utf-8")
        assert "including Analysis and Conclusion" in text
        assert "Never put figures that belong to different" in text
        assert "Do not recommend specific products" in text
        assert "actionable takeaways" not in text
    critic = (TEMPLATES / "research_verify_system.minijinja").read_text(encoding="utf-8")
    assert "combines figures" in critic and "recommends a product" in critic


# ── a fact-checker that gives no verdict (live bug: reasoning used the whole output budget) ──

async def test_no_verdict_from_the_fact_checker_publishes_the_report_with_a_note(config, redis, monkeypatch, no_scrape):
    async def search_kind_(redis_, config_, query, kind, allow_paid=True, recency=""):
        return [make_source(1, kind=kind)]

    class NoVerdictLLM(ScriptedLLM):
        async def __call__(self, config_, function_name, message, **kwargs):
            if function_name == "research_verify":
                self.calls.append(function_name)
                raise ValueError("TensorZero returned no text content: {'content': [{'type': 'thought'}]}")
            return await super().__call__(config_, function_name, message, **kwargs)

    llm = NoVerdictLLM({"sub_questions": [{"question": "Q", "queries": ["q"], "kinds": ["web"]}]}, [])
    monkeypatch.setattr(agents, "search_kind", search_kind_)
    monkeypatch.setattr(agents, "_tz_call", llm)
    config.search_max_rounds = 1
    final = await build_graph(config, redis).ainvoke(initial_state("topic", "s", [], ""))
    assert final["report"] and final["verified"] is False
    assert llm.calls.count("report_write") == 1          # no pointless rewrite: nothing was checked
    assert review_notes(final) == ["The automated fact-check could not complete for this draft, so its claims "
                                   "were not checked."]


async def test_other_critic_errors_still_fail_loudly(config, monkeypatch):
    async def broken(*args, **kwargs):
        raise ValueError("something else")

    monkeypatch.setattr(agents, "_tz_call", broken)
    with pytest.raises(ValueError):
        await agents.CriticAgent(config).run("report [1]", [{**make_source(1), "id": 1}])
