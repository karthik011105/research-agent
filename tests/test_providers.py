import asyncio
import httpx
import pytest
import respx
from app.tools import providers, search
from app.tools.providers import (
    arxiv_query, keyword_terms, openalex_abstract, parse_arxiv, parse_gdelt, parse_hn, parse_openalex,
    parse_semantic_scholar, top_comments,
)
from app.tools.search import SearchUnavailable, dedupe_new, interleave, search_kind
from tests.conftest import make_source

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2610.01234v1</id>
    <published>2026-10-02T17:59:59Z</published>
    <title>Layered Guardrails for
      Large Language Models</title>
    <summary>  We propose a layered guardrail system that blocks 97% of jailbreaks.  </summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
  </entry>
  <entry><title>No id</title><summary>x</summary></entry>
</feed>"""

S2_OK = {"data": [
    {"paperId": "abc", "title": "Guarding LLMs", "abstract": "A <i>survey</i> of guardrails.", "url": None,
     "year": 2026, "venue": "ACL", "citationCount": 12, "publicationDate": "2026-05-01"},
    {"paperId": "def", "title": "No abstract", "abstract": None, "url": "https://s2.org/def"},
]}

OPENALEX_OK = {"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": "OnionGuard: Layered Guardrails",
     "publication_date": "2026-02-05", "publication_year": 2026, "cited_by_count": 4,
     "abstract_inverted_index": {"We": [0], "propose": [1], "layered": [2], "guardrails.": [3]},
     "primary_location": {"landing_page_url": "https://zenodo.org/1", "source": {"display_name": "Zenodo"}}},
    {"id": "https://openalex.org/W2", "title": "OnionGuard: Layered Guardrails",      # second version of the same paper
     "abstract_inverted_index": {"Again": [0]}, "primary_location": {"landing_page_url": "https://zenodo.org/2"}},
    {"id": "https://openalex.org/W3", "title": "No abstract", "abstract_inverted_index": None},
]}

GDELT_OK = {"articles": [
    {"url": "https://news.example/a", "title": "Regulator fines AI firm", "seendate": "20261110T081500Z"},
    {"url": "", "title": "missing url"},
]}

HN_HIT = {"objectID": "42", "title": "Show HN: Open-source LLM guardrails", "url": "https://github.com/x/guard",
          "points": 210, "num_comments": 3, "created_at": "2026-09-30T10:00:00Z", "story_text": None}
HN_ITEM = {"children": [
    {"type": "comment", "text": "Works <b>well</b> in prod", "children": [{}, {}]},
    {"type": "comment", "text": "Latency is the issue", "children": []},
    {"type": "comment", "text": None, "children": []},
]}


def test_keyword_terms_drops_stopwords_and_short_words():
    assert keyword_terms("What are the latest LLM guardrails techniques in 2026?") == ["llm", "guardrails", "techniques", "2026"]
    assert arxiv_query("LLM guardrails") == "all:llm AND all:guardrails"


def test_parse_arxiv_reads_entries_and_skips_missing_id():
    [paper] = parse_arxiv(ARXIV_XML)
    assert paper["url"] == "https://arxiv.org/abs/2610.01234v1"
    assert paper["title"] == "Layered Guardrails for Large Language Models"
    assert paper["published_date"] == "2026-10-02"
    assert paper["content"].startswith("Authors: A. Author, B. Author. Abstract: We propose")
    assert (paper["provider"], paper["kind"]) == ("arxiv", "paper")


def test_parse_semantic_scholar_builds_url_and_skips_title_only():
    [paper] = parse_semantic_scholar(S2_OK)
    assert paper["url"] == "https://www.semanticscholar.org/paper/abc"
    assert paper["content"] == "2026; ACL; 12 citations. Abstract: A survey of guardrails."
    assert paper["kind"] == "paper"


def test_openalex_abstract_rebuilds_inverted_index():
    assert openalex_abstract({"b": [1, 3], "a": [0], "c": [2]}) == "a b c b"
    assert openalex_abstract(None) == ""


def test_parse_openalex_skips_duplicate_versions_and_missing_abstracts():
    [paper] = parse_openalex(OPENALEX_OK)
    assert paper["url"] == "https://zenodo.org/1"
    assert paper["content"] == "2026; Zenodo; 4 citations. Abstract: We propose layered guardrails."
    assert (paper["provider"], paper["kind"], paper["published_date"]) == ("openalex", "paper", "2026-02-05")


def test_parse_gdelt_is_headline_only_news():
    [article] = parse_gdelt(GDELT_OK)
    assert article["published_date"] == "2026-11-10"
    assert article["content"] == "Regulator fines AI firm"
    assert (article["provider"], article["kind"]) == ("gdelt", "news")


def test_parse_hn_uses_discussion_link_and_most_replied_comments():
    assert top_comments(HN_ITEM) == ["Works well in prod", "Latency is the issue"]
    src = parse_hn(HN_HIT, HN_ITEM)
    assert src["url"] == "https://news.ycombinator.com/item?id=42"
    assert "210 points, 3 comments" in src["content"] and "Linked article: https://github.com/x/guard" in src["content"]
    assert "Top comments: Works well in prod | Latency is the issue" in src["content"]
    assert src["kind"] == "community"


def test_interleave_and_dedupe_keep_every_list_represented():
    a = [make_source(1), make_source(2)]
    b = [make_source(3), make_source(1, url="https://example.com/1/")]
    assert [s["title"] for s in interleave([a, b])] == ["Page 1", "Page 3", "Page 2", "Page 1"]
    existing = [{**make_source(2), "id": 1}]
    assert [s["url"] for s in dedupe_new([a, b], existing)] == ["https://example.com/1", "https://example.com/3"]


def test_dedupe_matches_same_paper_under_different_urls():
    arxiv = {**make_source(1, url="https://arxiv.org/abs/1"), "title": "Layered Guardrails for LLMs"}
    doi = {**make_source(2, url="https://doi.org/10.1/x"), "title": "Layered guardrails for LLMs."}
    short = [{**make_source(3), "title": "AI"}, {**make_source(4), "title": "AI"}]   # short titles: URL only
    assert [s["url"] for s in dedupe_new([[arxiv], [doi], short], [])] == [
        "https://arxiv.org/abs/1", "https://example.com/3", "https://example.com/4"]


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    async def instant(provider):
        return None
    monkeypatch.setattr(providers, "throttle", instant)


@respx.mock
async def test_paper_search_uses_both_free_providers_without_spending_budget(config, redis):
    config.search_daily_limit = 0          # any paid call would raise
    respx.get(providers.OPENALEX_URL).mock(return_value=httpx.Response(200, json=OPENALEX_OK))
    arxiv = respx.get(providers.ARXIV_URL).mock(return_value=httpx.Response(200, text=ARXIV_XML))
    respx.get(providers.SEMANTIC_SCHOLAR_URL).mock(return_value=httpx.Response(200, json=S2_OK))
    results = await search_kind(redis, config, "llm guardrails", "paper")
    assert [r["provider"] for r in results] == ["openalex", "arxiv", "semantic_scholar"]
    await search_kind(redis, config, "llm guardrails", "paper")   # second call comes from cache
    assert arxiv.call_count == 1


@respx.mock
async def test_news_combines_tavily_and_gdelt_and_survives_one_failing(config, redis):
    respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(500))
    respx.get(providers.GDELT_URL).mock(return_value=httpx.Response(200, json=GDELT_OK))
    results = await search_kind(redis, config, "ai regulation fines", "news")
    assert [r["provider"] for r in results] == ["gdelt"]


@respx.mock
async def test_news_without_paid_budget_still_uses_gdelt(config, redis):
    tavily = respx.post(search._TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(providers.GDELT_URL).mock(return_value=httpx.Response(200, json=GDELT_OK))
    results = await search_kind(redis, config, "ai regulation", "news", allow_paid=False)
    assert tavily.call_count == 0 and results[0]["provider"] == "gdelt"


@respx.mock
async def test_gdelt_plain_text_error_counts_as_no_results(config, redis):
    respx.get(providers.GDELT_URL).mock(return_value=httpx.Response(200, text="Your search contained a keyword that was too short"))
    config.enabled_providers = ["gdelt"]
    assert await search_kind(redis, config, "ai", "news") == []


@respx.mock
async def test_rate_limited_free_provider_is_not_retried(config, redis):
    config.llm_max_retries = 3
    config.enabled_providers = ["semantic_scholar"]
    route = respx.get(providers.SEMANTIC_SCHOLAR_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(SearchUnavailable, match="All paper providers failed"):
        await search_kind(redis, config, "llm guardrails", "paper")
    assert route.call_count == 1


@respx.mock
async def test_rate_limit_puts_provider_on_cooldown_for_all_queries(config, redis):
    config.enabled_providers = ["arxiv"]
    route = respx.get(providers.ARXIV_URL).mock(return_value=httpx.Response(429, headers={"retry-after": "120"}))
    with pytest.raises(SearchUnavailable):
        await search_kind(redis, config, "llm guardrails", "paper")
    with pytest.raises(SearchUnavailable, match="All paper providers failed"):
        await search_kind(redis, config, "a different query", "paper")
    assert route.call_count == 1                       # second query never hit the API
    assert 0 < await redis.ttl("provider_cooldown:arxiv") <= 120


@respx.mock
async def test_calls_queued_behind_throttle_see_a_cooldown_set_meanwhile(config, redis, monkeypatch):
    """Production case: six arXiv queries queued on the 3 s throttle, the first got 429, and the other five
    still went out. With the cooldown checked after the throttle, only the first reaches the API."""
    config.enabled_providers = ["arxiv"]
    lock = asyncio.Lock()

    async def serial_throttle(provider):          # one call at a time, like the real throttle
        await lock.acquire()
        asyncio.get_running_loop().call_later(0.01, lock.release)

    monkeypatch.setattr(providers, "throttle", serial_throttle)
    route = respx.get(providers.ARXIV_URL).mock(return_value=httpx.Response(429))
    outcomes = await asyncio.gather(*(search_kind(redis, config, f"query {i}", "paper") for i in range(5)),
                                    return_exceptions=True)
    assert all(isinstance(o, SearchUnavailable) for o in outcomes)
    assert route.call_count == 1


@respx.mock
async def test_community_search_fetches_hn_comments(config, redis):
    respx.get(providers.HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={"hits": [HN_HIT, {**HN_HIT, "objectID": "7", "num_comments": 0}]}))
    respx.get(providers.HN_ITEM_URL.format(id="42")).mock(return_value=httpx.Response(200, json=HN_ITEM))
    [src] = await search_kind(redis, config, "llm guardrails", "community")
    assert "Top comments" in src["content"]


async def test_disabled_kind_raises(config, redis):
    config.enabled_providers = ["tavily"]
    with pytest.raises(SearchUnavailable, match="No provider enabled"):
        await search_kind(redis, config, "q", "paper")


async def test_throttle_spaces_calls(monkeypatch):
    monkeypatch.undo()                       # use the real throttle here
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(providers.asyncio, "sleep", fake_sleep)
    providers._last_call.pop("arxiv", None)
    await providers.throttle("arxiv")
    await providers.throttle("arxiv")
    assert len(waits) == 1 and 2.5 < waits[0] <= 3.0
