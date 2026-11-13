import json
import numpy as np
import pytest
from app import cache
from app.cache import cache_get, cache_set

# Tiny fake embedding space: each "meaning" is a unit vector, so tests control similarity exactly
VECTORS = {
    "ai chips": [1.0, 0.0, 0.0],
    "AI chips this month": [0.95, 0.312, 0.0],       # cosine 0.95 with "ai chips"
    "ai regulation": [0.6, 0.0, 0.8],                # cosine 0.6 with "ai chips": different topic
    "quantum": [0.0, 0.0, 1.0],
}


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    calls = []

    async def embed(text):
        calls.append(text)
        v = np.asarray(VECTORS[text])
        return (v / np.linalg.norm(v)).tolist()

    monkeypatch.setattr(cache, "embed", embed)
    return calls


@pytest.fixture
def ccfg(config):
    config.cache_ttl = 3600
    config.cache_similarity_threshold = 0.85
    config.cache_max_entries = 1000
    config.memory_scope = "global"
    return config


async def test_similar_question_hits_and_different_topic_misses(redis, ccfg):
    await cache_set(redis, ccfg, "ai chips", "chips report", [{"id": 1}])
    assert await cache_get(redis, ccfg, "AI chips this month") == {"report": "chips report", "sources": [{"id": 1}]}
    assert await cache_get(redis, ccfg, "ai regulation") is None


async def test_lookup_reads_the_index_once_not_every_key(redis, ccfg, monkeypatch):
    for topic in ("ai chips", "ai regulation", "quantum"):
        await cache_set(redis, ccfg, topic, f"{topic} report", [])
    gets = []
    real_get = redis.get

    async def counting_get(key):
        gets.append(key)
        return await real_get(key)

    monkeypatch.setattr(redis, "get", counting_get)
    assert (await cache_get(redis, ccfg, "AI chips this month"))["report"] == "ai chips report"
    assert len(gets) == 1               # only the winning entry is fetched (old code: one GET per cached key)


async def test_best_match_wins_over_older_weaker_match(redis, ccfg):
    ccfg.cache_similarity_threshold = 0.5
    await cache_set(redis, ccfg, "ai regulation", "regulation report", [])
    await cache_set(redis, ccfg, "ai chips", "chips report", [])
    assert (await cache_get(redis, ccfg, "AI chips this month"))["report"] == "chips report"


async def test_keys_are_stable_and_case_insensitive(redis, ccfg):
    await cache_set(redis, ccfg, "ai chips", "v1", [])
    assert cache._entry_key("AI   Chips") == cache._entry_key("ai chips")    # same entry, not a duplicate
    assert await redis.hlen("semantic:index") == 1


async def test_expired_entries_are_pruned_from_the_index(redis, ccfg, monkeypatch):
    await cache_set(redis, ccfg, "ai chips", "old", [])
    await redis.delete(cache._entry_key("ai chips"))           # report key expired (TTL)
    assert await cache_get(redis, ccfg, "AI chips this month") is None
    assert await redis.hlen("semantic:index") == 0

    await cache_set(redis, ccfg, "quantum", "q", [])
    monkeypatch.setattr(cache.time, "time", lambda: 10**12)    # far future: index entry itself expired
    assert await cache_get(redis, ccfg, "quantum") is None
    assert await redis.hlen("semantic:index") == 0


async def test_index_size_is_bounded(redis, ccfg):
    ccfg.cache_max_entries = 2
    for topic in ("ai chips", "ai regulation", "quantum"):
        await cache_set(redis, ccfg, topic, topic, [])
    assert await redis.hlen("semantic:index") == 2
    assert await redis.hget("semantic:index", cache._entry_key("quantum")) is not None   # newest kept


async def test_corrupt_index_entry_is_ignored(redis, ccfg):
    await redis.hset("semantic:index", "semantic:junk", "not json")
    await cache_set(redis, ccfg, "ai chips", "chips report", [])
    assert (await cache_get(redis, ccfg, "ai chips"))["report"] == "chips report"
    assert await redis.hget("semantic:index", "semantic:junk") is None


async def test_session_scope_hides_other_sessions_reports(redis, ccfg):
    ccfg.memory_scope = "session"
    await cache_set(redis, ccfg, "ai chips", "attacker's poisoned report", [], session_id="attacker")
    assert await cache_get(redis, ccfg, "AI chips this month", session_id="victim") is None
    assert (await cache_get(redis, ccfg, "AI chips this month", session_id="attacker"))["report"] == "attacker's poisoned report"
    ccfg.memory_scope = "global"                               # default: shared by everyone (old behaviour)
    await cache_set(redis, ccfg, "quantum", "shared", [], session_id="someone")
    assert (await cache_get(redis, ccfg, "quantum", session_id="anyone"))["report"] == "shared"


async def test_size_cap_counts_every_sessions_entries(redis, ccfg):
    ccfg.memory_scope, ccfg.cache_max_entries = "session", 2
    for i, topic in enumerate(("ai chips", "ai regulation", "quantum")):
        await cache_set(redis, ccfg, topic, topic, [], session_id=f"s{i}")
    assert await redis.hlen("semantic:index") == 2
