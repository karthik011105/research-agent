import asyncio
import hashlib
import json
import logging
from datetime import date
import httpx
import redis.asyncio as aioredis
from langsmith import traceable
from app.config import Config
from app.retry import with_retry

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_CACHE_PREFIX = "search:"
_BUDGET_PREFIX = "search_budget:"
_MIN_CONTENT_CHARS = 80


class SearchUnavailable(Exception):
    """No provider configured, daily budget used up, or every provider failed."""


def _source(title: str, url: str, content: str, published_date: str, provider: str) -> dict:
    return {
        "title": (title or url).strip(),
        "url": url.strip(),
        "content": " ".join((content or "").split()),
        "published_date": published_date or "",
        "provider": provider,
    }


def parse_tavily(payload: dict) -> list[dict]:
    return [
        _source(r.get("title", ""), r.get("url", ""), r.get("content", ""), r.get("published_date", ""), "tavily")
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
        results.append(_source(r.get("title", ""), r["url"], content, r.get("page_age", ""), "brave"))
    return results


async def _tavily(config: Config, query: str, topic: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            _TAVILY_URL,
            headers={"Authorization": f"Bearer {config.tavily_api_key}"},
            json={
                "query": query,
                "topic": topic,
                "search_depth": "basic",
                "max_results": config.search_max_results,
            },
        )
        r.raise_for_status()
        return parse_tavily(r.json())


async def _brave(config: Config, query: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            _BRAVE_URL,
            headers={"X-Subscription-Token": config.brave_api_key, "Accept": "application/json"},
            params={"q": query, "count": config.search_max_results},
        )
        r.raise_for_status()
        return parse_brave(r.json())


async def _take_budget(redis: aioredis.Redis, config: Config) -> bool:
    key = f"{_BUDGET_PREFIX}{date.today().isoformat()}"
    used = await redis.incr(key)
    if used == 1:
        await redis.expire(key, 86400 * 2)
    return used <= config.search_daily_limit


def _cache_key(query: str, topic: str) -> str:
    return _CACHE_PREFIX + hashlib.sha256(f"{topic}|{query.strip().lower()}".encode()).hexdigest()


@traceable(run_type="tool", name="tool:web_search")
async def web_search(redis: aioredis.Redis, config: Config, query: str, topic: str = "general") -> list[dict]:
    """One query → normalized results. Cached per query; counts against the daily budget on a miss."""
    key = _cache_key(query, topic)
    cached = await redis.get(key)
    if cached:
        return json.loads(cached)

    providers = []
    if config.tavily_api_key:
        providers.append(("tavily", lambda: _tavily(config, query, topic)))
    if config.brave_api_key:
        providers.append(("brave", lambda: _brave(config, query)))
    if not providers:
        raise SearchUnavailable("No search API key configured")
    if not await _take_budget(redis, config):
        raise SearchUnavailable(f"Daily search limit of {config.search_daily_limit} reached")

    for name, call in providers:
        try:
            results = await with_retry(call, max_retries=config.llm_max_retries, delay=config.llm_retry_delay)
            await redis.setex(key, config.search_cache_ttl, json.dumps(results))
            return results
        except Exception as e:
            logger.warning(f"Search provider {name} failed for '{query}': {e}")
    raise SearchUnavailable("All search providers failed")


def merge_sources(result_lists: list[list[dict]], existing: list[dict], config: Config) -> list[dict]:
    """Dedupe by URL, drop near-empty pages, truncate content, number sources 1..N (existing ids are kept)."""
    seen = {s["url"].rstrip("/") for s in existing}
    merged = list(existing)
    for results in result_lists:
        for s in results:
            url = s["url"].rstrip("/")
            if url in seen or len(s["content"]) < _MIN_CONTENT_CHARS:
                continue
            if len(merged) >= config.search_max_sources:
                return merged
            seen.add(url)
            merged.append({**s, "id": len(merged) + 1, "content": s["content"][:config.source_char_limit]})
    return merged


async def search_many(redis: aioredis.Redis, config: Config, queries: list[str], existing: list[dict], topic: str = "general") -> list[dict]:
    """Runs queries in parallel. Individual query failures are skipped; raises only if nothing came back at all."""
    if not queries:
        return existing
    outcomes = await asyncio.gather(*(web_search(redis, config, q, topic) for q in queries), return_exceptions=True)
    ok = [o for o in outcomes if not isinstance(o, Exception)]
    for q, o in zip(queries, outcomes):
        if isinstance(o, Exception):
            logger.warning(f"Query '{q}' failed: {o}")
    if not ok and not existing:
        first_error = next(o for o in outcomes if isinstance(o, Exception))
        raise SearchUnavailable(str(first_error))
    return merge_sources(ok, existing, config)


def format_sources(sources: list[dict]) -> str:
    """Numbered, delimited source blocks for prompts. The delimiters mark web text as untrusted data."""
    blocks = []
    for s in sources:
        title = s["title"].replace('"', "'")
        meta = f'id="{s["id"]}" title="{title}" url="{s["url"]}"'
        if s.get("published_date"):
            meta += f' published="{s["published_date"]}"'
        blocks.append(f"<source {meta}>\n{s['content']}\n</source>")
    return "\n\n".join(blocks)
