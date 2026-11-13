import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import date, timedelta
from urllib.parse import urlsplit
import httpx
import redis.asyncio as aioredis
from langsmith import traceable
from app import metering
from app.config import Config
from app.retry import with_retry
from app.tools import providers
from app.tools.providers import make_source

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_CACHE_PREFIX = "search:"
_BUDGET_PREFIX = "search_budget:"
_MIN_CONTENT_CHARS = 80


KINDS = ("web", "news", "paper", "community")
# Time window a topic asks for ("what's new today" -> week). The planner chooses it; "" = no date limit.
# "today" maps to a week: free news sources are too thin for a single day.
RECENCY_DAYS = {"week": 7, "month": 31, "year": 365}
_BRAVE_FRESHNESS = {"week": "pw", "month": "pm", "year": "py"}
# Papers move slower than news: a "latest" question still takes papers from the past year
_PAPER_MIN_DAYS = 365


def recency_window(recency: str, kind: str) -> int:
    """Days of history a source of this kind may be from, for a topic's recency (0 = no limit)."""
    days = RECENCY_DAYS.get(recency, 0)
    return max(days, _PAPER_MIN_DAYS) if days and kind == "paper" else days


def is_stale(source: dict, recency: str, today: date) -> bool:
    """True if the source is dated and clearly older than the topic asks for. Undated pages are kept, and the
    window is doubled because pages are often dated by first publication while still being updated."""
    days = recency_window(recency, source.get("kind", "web"))
    if not days:
        return False
    try:
        published = date.fromisoformat((source.get("published_date") or "")[:10])
    except ValueError:
        return False
    return (today - published).days > 2 * days


def prefer_recent(sources: list[dict], recency: str, today: date, min_keep: int = 3) -> tuple[list[dict], int]:
    """Drops stale sources for a time-sensitive topic, unless that leaves fewer than min_keep: then the stale
    ones stay, after the recent ones, so a thin day still gets a report. Returns (sources, number dropped)."""
    if not RECENCY_DAYS.get(recency):
        return sources, 0
    recent = [s for s in sources if not is_stale(s, recency, today)]
    if len(recent) >= min(min_keep, len(sources)):
        return recent, len(sources) - len(recent)
    return recent + [s for s in sources if is_stale(s, recency, today)], 0
# Kinds whose providers spend the paid web-search budget (Tavily/Brave); the others are free APIs
BUDGETED_KINDS = {"web", "news"}


class SearchUnavailable(Exception):
    """No provider configured, daily budget used up, or every provider failed."""


def parse_tavily(payload: dict, kind: str = "web") -> list[dict]:
    return [
        make_source(r.get("title", ""), r.get("url", ""), r.get("content", ""), r.get("published_date", ""), "tavily", kind)
        for r in payload.get("results", [])
        if r.get("url")
    ]


def parse_brave(payload: dict) -> list[dict]:
    results = []
    for r in payload.get("web", {}).get("results", []):
        if not r.get("url"):
            continue
        # Brave returns a short description plus optional extra snippets — join them for more context
        content = " ".join([r.get("description", "")] + r.get("extra_snippets", []))
        results.append(make_source(r.get("title", ""), r["url"], content, r.get("page_age", ""), "brave", "web"))
    return results


async def _tavily(config: Config, query: str, topic: str, recency: str = "") -> list[dict]:
    body = {"query": query, "topic": topic, "search_depth": "basic", "max_results": config.search_max_results}
    if recency in RECENCY_DAYS:
        body["time_range"] = recency
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            _TAVILY_URL,
            headers={"Authorization": f"Bearer {config.tavily_api_key}"},
            json=body,
        )
        r.raise_for_status()
        return parse_tavily(r.json(), "news" if topic == "news" else "web")


async def _brave(config: Config, query: str, recency: str = "") -> list[dict]:
    params = {"q": query, "count": config.search_max_results}
    if recency in _BRAVE_FRESHNESS:
        params["freshness"] = _BRAVE_FRESHNESS[recency]
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            _BRAVE_URL,
            headers={"X-Subscription-Token": config.brave_api_key, "Accept": "application/json"},
            params=params,
        )
        r.raise_for_status()
        return parse_brave(r.json())


async def _take_budget(redis: aioredis.Redis, config: Config) -> bool:
    key = f"{_BUDGET_PREFIX}{date.today().isoformat()}"
    used = await redis.incr(key)
    if used == 1:
        await redis.expire(key, 86400 * 2)
    return used <= config.search_daily_limit


def _cache_key(query: str, topic: str, recency: str = "") -> str:
    # A date-limited search is a different search: it must not be answered from an unlimited one's cache
    scope = f"{topic}@{recency}" if recency else topic
    return _CACHE_PREFIX + hashlib.sha256(f"{scope}|{query.strip().lower()}".encode()).hexdigest()


@traceable(run_type="tool", name="tool:web_search")
async def web_search(redis: aioredis.Redis, config: Config, query: str, topic: str = "general",
                     recency: str = "") -> list[dict]:
    """One query → normalized results. Cached per query; counts against the daily budget on a miss."""
    key = _cache_key(query, topic, recency)
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)

    providers = []
    if config.tavily_api_key:
        providers.append(("tavily", lambda: _tavily(config, query, topic, recency)))
    if config.brave_api_key:
        providers.append(("brave", lambda: _brave(config, query, recency)))
    if not providers:
        raise SearchUnavailable("No search API key configured")
    if not await _take_budget(redis, config):
        raise SearchUnavailable(f"Daily search limit of {config.search_daily_limit} reached")

    metering.record(paid_searches=1)
    for name, call in providers:
        try:
            results = await with_retry(call, max_retries=config.llm_max_retries, delay=config.llm_retry_delay)
            await redis.setex(key, config.search_cache_ttl, json.dumps(results))
            return results
        except Exception as e:
            logger.warning(f"Search provider {name} failed for '{query}': {e}")
    raise SearchUnavailable("All search providers failed")


_COOLDOWN_PREFIX = "provider_cooldown:"
_DEFAULT_COOLDOWN = 300


def _is_rate_limited(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _retry_after(exc: httpx.HTTPStatusError) -> int:
    try:
        seconds = int(exc.response.headers.get("retry-after", ""))
    except ValueError:
        seconds = _DEFAULT_COOLDOWN
    return min(max(seconds, 30), 3600)


async def _free_search(redis: aioredis.Redis, config: Config, provider: str, query: str, fetch,
                       recency: str = "") -> list[dict]:
    """Keyless providers: cached like web search, never counted against the paid daily budget.
    A 429 is not retried; instead the provider is skipped by every job until its cooldown ends, so a
    rate-limited API costs one failed call rather than slowing down each search."""
    key = _cache_key(query, provider, recency)
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)
    cooldown_key = f"{_COOLDOWN_PREFIX}{provider}"

    async def attempt():
        # Wait for the provider's rate slot first, then check the cooldown: a call that queued behind
        # another one must see a 429 that arrived while it was waiting.
        await providers.throttle(provider)
        if await redis.exists(cooldown_key):
            raise SearchUnavailable(f"{provider} is cooling down after a rate limit")
        metering.record(free_searches=1)
        return await fetch()

    try:
        results = await with_retry(attempt, max_retries=config.llm_max_retries, delay=config.llm_retry_delay,
                                   retry_if=lambda e: not (_is_rate_limited(e) or isinstance(e, SearchUnavailable)))
    except httpx.HTTPStatusError as e:
        if _is_rate_limited(e):
            await redis.setex(cooldown_key, _retry_after(e), "1")
        raise
    await redis.setex(key, config.search_cache_ttl, json.dumps(results))
    return results


def _providers_for(redis: aioredis.Redis, config: Config, query: str, kind: str, allow_paid: bool,
                   recency: str = "") -> list[tuple[str, object]]:
    enabled = set(config.enabled_providers)
    n = config.search_max_results
    web = allow_paid and bool(enabled & {"tavily", "brave"})
    days = recency_window(recency, kind)
    since = (date.today() - timedelta(days=days)).isoformat() if days else ""

    def free(provider, fetch):
        return lambda: _free_search(redis, config, provider, query, fetch, recency)

    calls = []
    if kind == "web" and web:
        calls.append(("web", lambda: web_search(redis, config, query, "general", recency)))
    elif kind == "news":
        if web:
            calls.append(("web", lambda: web_search(redis, config, query, "news", recency)))
        if "gdelt" in enabled:
            span = "1w" if recency == "week" else "1m"   # GDELT keeps about three months of articles
            calls.append(("gdelt", free("gdelt", lambda: providers.fetch_gdelt(query, n, span))))
    elif kind == "paper":
        if "openalex" in enabled:
            calls.append(("openalex", free("openalex", lambda: providers.fetch_openalex(query, n, since))))
        if "arxiv" in enabled:
            calls.append(("arxiv", free("arxiv", lambda: providers.fetch_arxiv(query, n))))
        if "semantic_scholar" in enabled:
            calls.append(("semantic_scholar", free("semantic_scholar", lambda: providers.fetch_semantic_scholar(
                query, n, config.semantic_scholar_api_key))))
    elif kind == "community" and "hackernews" in enabled:
        since_ts = int(time.time()) - days * 86400 if days else 0
        calls.append(("hackernews", free("hackernews", lambda: providers.fetch_hackernews(query, min(n, 3), since_ts))))
    return calls


def interleave(result_lists: list[list[dict]]) -> list[dict]:
    """Round-robin across lists so every provider / sub-question keeps its best results when sources are capped."""
    out = []
    for i in range(max((len(r) for r in result_lists), default=0)):
        out.extend(r[i] for r in result_lists if i < len(r))
    return out


@traceable(run_type="tool", name="tool:search_kind")
async def search_kind(redis: aioredis.Redis, config: Config, query: str, kind: str, allow_paid: bool = True,
                      recency: str = "") -> list[dict]:
    """One query against every enabled provider for a kind of source (web, news, paper, community), in parallel.
    allow_paid=False skips Tavily/Brave (the per-job web budget is spent). recency ("week", "month", "year")
    asks each provider for recent results only. Individual providers may fail; raises SearchUnavailable only
    when none of them returned anything."""
    calls = _providers_for(redis, config, query, kind, allow_paid, recency)
    if not calls:
        raise SearchUnavailable(f"No provider enabled for '{kind}' sources")
    outcomes = await asyncio.gather(*(call() for _, call in calls), return_exceptions=True)
    ok = []
    for (name, _), outcome in zip(calls, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(f"{name} failed for '{query}' ({kind}): {outcome}")
        else:
            ok.append(outcome)
    if not ok:
        raise SearchUnavailable(f"All {kind} providers failed: {outcomes[0]}")
    return interleave(ok)


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def is_blocked(url: str, blocked_domains: list[str]) -> bool:
    """True if the URL's host is a blocked domain or one of its subdomains (e.g. news.polymarket.com)."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in blocked_domains)


def drop_blocked(sources: list[dict], blocked_domains: list[str]) -> list[dict]:
    return [s for s in sources if not is_blocked(s["url"], blocked_domains)] if blocked_domains else sources


def dedupe_new(result_lists: list[list[dict]], existing: list[dict]) -> list[dict]:
    """Interleaved results that are not already a source, first occurrence wins. Matches on URL and on
    title, because the same paper or article often appears under several URLs (arXiv, DOI, publisher)."""
    seen_urls = {s["url"].rstrip("/") for s in existing}
    seen_titles = {_title_key(s["title"]) for s in existing}
    fresh = []
    for s in interleave(result_lists):
        url, title = s["url"].rstrip("/"), _title_key(s["title"])
        if url in seen_urls or (len(title) > 15 and title in seen_titles):
            continue
        seen_urls.add(url)
        seen_titles.add(title)
        fresh.append(s)
    return fresh


def merge_sources(result_lists: list[list[dict]], existing: list[dict], config: Config,
                  limit: int | None = None) -> list[dict]:
    """Dedupe by URL, drop near-empty pages, truncate content, number sources 1..N (existing ids are kept).
    limit (default search_max_sources) caps the total; the first search round passes a lower one to keep
    room for the gap-check round."""
    limit = config.search_max_sources if limit is None else limit
    seen = {s["url"].rstrip("/") for s in existing}
    merged = list(existing)
    for results in result_lists:
        for s in results:
            url = s["url"].rstrip("/")
            if url in seen or len(s["content"]) < _MIN_CONTENT_CHARS:
                continue
            if len(merged) >= limit:
                return merged
            seen.add(url)
            merged.append({**s, "id": len(merged) + 1, "content": s["content"][:config.source_char_limit]})
    return merged


def format_sources(sources: list[dict]) -> str:
    """Numbered, delimited source blocks for prompts. The delimiters mark web text as untrusted data."""
    blocks = []
    for s in sources:
        title = s["title"].replace('"', "'")
        meta = f'id="{s["id"]}" title="{title}" url="{s["url"]}"'
        if s.get("kind"):
            meta += f' type="{s["kind"]}"'
        if s.get("published_date"):
            meta += f' published="{s["published_date"]}"'
        blocks.append(f"<source {meta}>\n{s['content']}\n</source>")
    return "\n\n".join(blocks)
