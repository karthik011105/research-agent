from types import SimpleNamespace
import fakeredis.aioredis
import pytest


@pytest.fixture
def config():
    """Stand-in for app.config.Config (the real one reads AWS Secrets Manager)."""
    return SimpleNamespace(
        tavily_api_key="tvly-test",
        brave_api_key="",
        search_max_results=5,
        search_max_sources=8,
        search_max_queries=4,
        search_max_rounds=2,
        search_daily_limit=40,
        search_cache_ttl=3600,
        source_char_limit=1500,
        llm_max_retries=1,
        llm_retry_delay=0.0,
        llm_rate_limit_retries=2,
        critic_source_chars=600,
        agent_report_truncate=3000,
        critic_report_truncate=8000,
        agent_max_iterations=2,
        tensorzero_url="http://tensorzero.test",
    )


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def make_source(n: int, url: str | None = None, content: str | None = None) -> dict:
    return {
        "title": f"Page {n}",
        "url": url or f"https://example.com/{n}",
        "content": content if content is not None else f"Useful fact number {n}. " * 10,
        "published_date": "",
        "provider": "tavily",
    }
