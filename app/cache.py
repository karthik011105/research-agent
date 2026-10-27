import json
import numpy as np
import redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer
from app.config import Config

_model = SentenceTransformer("all-MiniLM-L6-v2")
_CACHE_PREFIX = "semantic:"
_EMB_PREFIX = "emb:"


def _cosine_similarity(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def _embed(text: str) -> list:
    return _model.encode(text).tolist()


async def cache_get(redis: aioredis.Redis, config: Config, query: str) -> dict | None:
    """Returns {"report", "sources"} for a semantically similar earlier query."""
    query_emb = _embed(query)
    async for key in redis.scan_iter(f"{_EMB_PREFIX}*"):
        stored = await redis.get(key)
        if stored is None:
            continue
        if _cosine_similarity(query_emb, json.loads(stored)) >= config.cache_similarity_threshold:
            cached = await redis.get(key.replace(_EMB_PREFIX, _CACHE_PREFIX))
            if cached is None:
                continue
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                # Entry written before sources were cached: plain report text
                return {"report": cached, "sources": []}
    return None


async def cache_set(redis: aioredis.Redis, config: Config, query: str, report: str, sources: list[dict]) -> None:
    key_suffix = abs(hash(query))
    await redis.setex(f"{_CACHE_PREFIX}{key_suffix}", config.cache_ttl, json.dumps({"report": report, "sources": sources}))
    await redis.setex(f"{_EMB_PREFIX}{key_suffix}", config.cache_ttl, json.dumps(_embed(query)))
