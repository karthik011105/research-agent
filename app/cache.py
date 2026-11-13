"""Semantic report cache: a new question close enough in meaning to a recent one reuses its report.

Layout in Redis:
  semantic:index          hash  {entry key -> {"e": embedding, "x": expiry unix time}}
  semantic:<sha256>       string {"report", "sources"}  with TTL = cache_ttl

A lookup is one HGETALL plus one matrix-vector product. The previous version issued a SCAN plus one GET
per cached entry on every request (and used Python's per-process hash() for keys, so a restart created
duplicates). Hash fields can't expire on their own, so stale index entries are pruned on read and write.
"""
import hashlib
import json
import time
import numpy as np
import redis.asyncio as aioredis
from app.config import Config
from app.embeddings import embed

_INDEX = "semantic:index"
_ENTRY_PREFIX = "semantic:"


def _entry_key(query: str) -> str:
    return _ENTRY_PREFIX + hashlib.sha256(" ".join(query.lower().split()).encode()).hexdigest()


def _parse_index(raw: dict, now: float) -> tuple[list[str], list[list[float]], list[str]]:
    """(live keys, their embeddings, expired or unreadable keys)."""
    keys, vectors, stale = [], [], []
    for key, value in raw.items():
        try:
            item = json.loads(value)
            if item["x"] <= now:
                stale.append(key)
                continue
            keys.append(key)
            vectors.append(item["e"])
        except (ValueError, KeyError, TypeError):
            stale.append(key)
    return keys, vectors, stale


async def cache_get(redis: aioredis.Redis, config: Config, query: str) -> dict | None:
    """Returns {"report", "sources"} for the most similar recent query above the threshold, else None."""
    raw = await redis.hgetall(_INDEX)
    if not raw:
        return None
    keys, vectors, stale = _parse_index(raw, time.time())
    if stale:
        await redis.hdel(_INDEX, *stale)
    if not keys:
        return None

    query_vec = np.asarray(await embed(query))
    similarities = np.asarray(vectors) @ query_vec          # embeddings are unit length: dot = cosine
    for i in np.argsort(-similarities):
        if similarities[i] < config.cache_similarity_threshold:
            break
        cached = await redis.get(keys[i])
        if cached is None:                                   # entry expired between index and read
            await redis.hdel(_INDEX, keys[i])
            continue
        return json.loads(cached)
    return None


async def cache_set(redis: aioredis.Redis, config: Config, query: str, report: str, sources: list[dict]) -> None:
    key = _entry_key(query)
    vector = await embed(query)
    expiry = time.time() + config.cache_ttl
    async with redis.pipeline(transaction=True) as pipe:
        pipe.setex(key, config.cache_ttl, json.dumps({"report": report, "sources": sources}))
        pipe.hset(_INDEX, key, json.dumps({"e": vector, "x": expiry}))
        await pipe.execute()
    await _trim_index(redis, config)


async def _trim_index(redis: aioredis.Redis, config: Config) -> None:
    """Keeps the index bounded: drop expired entries, then the soonest-to-expire beyond cache_max_entries."""
    if await redis.hlen(_INDEX) <= config.cache_max_entries:
        return
    raw = await redis.hgetall(_INDEX)
    keys, _, stale = _parse_index(raw, time.time())
    expiries = {k: json.loads(raw[k])["x"] for k in keys}
    overflow = sorted(keys, key=expiries.get)[:max(0, len(keys) - config.cache_max_entries)]
    drop = stale + overflow
    if drop:
        await redis.hdel(_INDEX, *drop)
