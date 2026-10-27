import asyncio
import logging

logger = logging.getLogger(__name__)


async def with_retry(coro_fn, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, retry_if=None):
    """Call coro_fn() with exponential backoff. Raises the last exception if all retries fail,
    or straight away when retry_if(exc) is False."""
    last_exc = None
    wait = delay
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            if retry_if is not None and not retry_if(exc):
                raise
            last_exc = exc
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}. Retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
                wait *= backoff
    raise last_exc
