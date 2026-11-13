import json
import httpx
import pytest
import respx
from app.tools import search
from app.tools.search import (
    SearchUnavailable, format_sources, merge_sources, parse_brave, parse_tavily, web_search,
)
from tests.conftest import make_source

TAVILY_OK = {
    "results": [
        {"title": "AI chips 2026", "url": "https://a.com/1", "content": "Market grew " * 20, "score": 0.9, "published_date": None},
        {"title": None, "url": "https://b.com/2", "content": "Other " * 20, "score": 0.5},
        {"title": "no url", "url": "", "content": "x"},
    ]
}


def test_parse_tavily_handles_null_fields_and_missing_url():
    results = parse_tavily(TAVILY_OK)
    assert [r["url"] for r in results] == ["https://a.com/1", "https://b.com/2"]
    assert results[0]["published_date"] == ""
    assert results[1]["title"] == "https://b.com/2"  # falls back to URL
    assert results[0]["provider"] == "tavily"


def test_parse_brave_joins_snippets():
    payload = {"web": {"results": [{"title": "T", "url": "https://c.com", "description": "desc", "extra_snippets": ["more"], "page_age": "2026-10-01"}]}}
    [r] = parse_brave(payload)
    assert r["content"] == "desc more"
    assert r["published_date"] == "2026-10-01"
    assert r["provider"] == "brave"


def test_merge_sources_dedupes_filters_truncates_and_numbers(config):
    config.source_char_limit = 50
    existing = [{**make_source(1), "id": 1}]
    new = [
        make_source(1, url="https://example.com/1/"),  # same page as existing (trailing slash)
        make_source(2, content="too short"),
        make_source(3),
        make_source(3),                                # duplicate in the same batch
        make_source(4),
    ]
    merged = merge_sources([new], existing, config)
    assert [s["id"] for s in merged] == [1, 2, 3]
    assert [s["url"] for s in merged] == ["https://example.com/1", "https://example.com/3", "https://example.com/4"]
    assert all(len(s["content"]) <= 50 for s in merged[1:])


def test_merge_sources_respects_max(config):
    config.search_max_sources = 2
    merged = merge_sources([[make_source(i) for i in range(5)]], [], config)
    assert len(merged) == 2


def test_format_sources_escapes_title_quotes():
    out = format_sources([{**make_source(1), "id": 1, "title": 'He said "hi"'}])
    assert 'title="He said \'hi\'"' in out
    assert out.startswith('<source id="1"') and out.endswith("</source>")


@respx.mock
async def test_web_search_calls_tavily_with_bearer_and_caches(config, redis):
    route = respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(200, json=TAVILY_OK))
    first = await web_search(redis, config, "ai chips", "news")
    second = await web_search(redis, config, "AI chips ", "news")  # same query after normalizing
    assert first == second
    assert route.call_count == 1
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer tvly-test"
    assert json.loads(request.content)["topic"] == "news"


@respx.mock
async def test_web_search_falls_back_to_brave(config, redis):
    config.brave_api_key = "brave-test"
    respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(500))
    respx.get(search._BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": [
        {"title": "B", "url": "https://brave.com/x", "description": "from brave " * 20}
    ]}}))
    results = await web_search(redis, config, "q")
    assert results[0]["provider"] == "brave"


async def test_web_search_without_keys_raises(config, redis):
    config.tavily_api_key = ""
    with pytest.raises(SearchUnavailable, match="No search API key"):
        await web_search(redis, config, "q")


@respx.mock
async def test_daily_budget_blocks_extra_searches(config, redis):
    config.search_daily_limit = 1
    respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(200, json=TAVILY_OK))
    await web_search(redis, config, "first")
    with pytest.raises(SearchUnavailable, match="Daily search limit"):
        await web_search(redis, config, "second")
    # cached queries don't spend budget
    assert await web_search(redis, config, "first")


def test_blocklist_matches_domain_and_subdomains_only():
    from app.tools.search import drop_blocked, is_blocked
    blocked = ["polymarket.com", "kalshi.com"]
    assert is_blocked("https://polymarket.com/event/us-enacts-ai-safety-bill", blocked)
    assert is_blocked("https://news.Polymarket.com/x", blocked)
    assert not is_blocked("https://notpolymarket.com/x", blocked)       # suffix must be a whole domain
    assert not is_blocked("https://reuters.com/polymarket.com", blocked)  # path doesn't count
    sources = [make_source(1, url="https://polymarket.com/e"), make_source(2)]
    assert [s["url"] for s in drop_blocked(sources, blocked)] == ["https://example.com/2"]
    assert drop_blocked(sources, []) == sources
