"""One shared sentence-embedding model for the semantic cache and long-term memory.

Before, cache.py and memory.py each loaded their own copy at import time (~100 MB each), and the cache
encoded on the event loop, briefly freezing every other request and job. The model now loads once, on first
use, and encoding runs in a worker thread. Vectors are L2-normalised, so cosine similarity is a dot product.
"""
import asyncio
import threading

MODEL_NAME = "all-MiniLM-L6-v2"   # baked into the app image at build time (see app/Dockerfile)

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer   # heavy import, only when first needed
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_sync(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


async def embed(text: str) -> list[float]:
    """Embedding for `text`, computed off the event loop."""
    return await asyncio.to_thread(embed_sync, text)
