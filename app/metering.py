"""Per-run usage counters (LLM calls, tokens, searches) for the benchmark.

A ContextVar holds one dict per run. asyncio tasks copy the context when they are created, and a copied
context still points at the same dict, so calls made inside LangGraph branches and asyncio.gather all add
to the run's totals. Outside a metered run, record() does nothing.
"""
from contextvars import ContextVar

_counters: ContextVar[dict | None] = ContextVar("bench_meter", default=None)


def start() -> dict:
    """Begin metering in the current context; returns the live counters dict."""
    counters = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "paid_searches": 0, "free_searches": 0,
                "models": {}}
    _counters.set(counters)
    return counters


def active() -> bool:
    return _counters.get() is not None


def record_model(name: str) -> None:
    """Counts which model actually served a call (TensorZero may fall back to another one on rate limits)."""
    counters = _counters.get()
    if counters is not None and name:
        counters["models"][name] = counters["models"].get(name, 0) + 1


def record(**amounts: int) -> None:
    counters = _counters.get()
    if counters is None:
        return
    for key, value in amounts.items():
        counters[key] = counters.get(key, 0) + (value or 0)
