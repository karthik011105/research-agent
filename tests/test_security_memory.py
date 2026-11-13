import pytest
from app.security.memory_poisoning import DEFENSE_SETTINGS, MemoryOps, run_memory_experiment

SIMILAR = {("crispr gene therapy results", "crispr therapy clinical results"): 0.95,
           ("crispr gene therapy results", "gene editing ethics"): 0.75}


class FakeMemory:
    """In-memory cache + LTM with the same owner/session semantics as app.cache / app.memory."""

    def __init__(self):
        self.cache, self.ltm = {}, {}

    def _sim(self, a, b):
        return 1.0 if a == b else SIMILAR.get((a, b), SIMILAR.get((b, a), 0.1))

    def _owner(self, config, session):
        return session if config.memory_scope == "session" else ""

    async def cache_get(self, redis, config, topic, session=None):
        owner = self._owner(config, session)
        for (t, o), report in self.cache.items():
            if o == owner and self._sim(t, topic) >= 0.85:
                return {"report": report, "sources": []}
        return None

    async def cache_set(self, redis, config, topic, report, sources, session=None):
        self.cache[(topic, self._owner(config, session))] = report

    async def cache_delete(self, redis, config, topic, session=None):
        self.cache.pop((topic, self._owner(config, session)), None)

    async def ltm_search(self, config, topic, session=None):
        owner = self._owner(config, session)
        return next(({"report": r} for (t, o, _), r in self.ltm.items() if o == owner and self._sim(t, topic) > 0.88), None)

    async def ltm_store(self, config, topic, report, report_id, sources, session=None):
        self.ltm[(topic, self._owner(config, session), report_id)] = report

    async def ltm_search_related(self, config, topic, session=None):
        owner = self._owner(config, session)
        return next((r for (t, o, _), r in self.ltm.items() if o == owner and 0.65 <= self._sim(t, topic) < 0.87), None)

    async def ltm_delete(self, report_id):
        self.ltm = {k: v for k, v in self.ltm.items() if k[2] != report_id}

    def ops(self):
        return MemoryOps(self.cache_get, self.cache_set, self.cache_delete, self.ltm_search, self.ltm_store,
                         self.ltm_search_related, self.ltm_delete)


VICTIMS = [("same", "crispr therapy clinical results"), ("related", "gene editing ethics"),
           ("unrelated", "ev sales trends")]


async def writer_that_copies_background(topic, background):
    return f"Report on {topic}. Background says: {background[:400]}"


@pytest.mark.parametrize("setting, verified, expected", [
    ("none", False, {"same": 1.0, "related": 1.0, "unrelated": 0.0}),
    ("verified_only", False, {"same": 0.0, "related": 0.0, "unrelated": 0.0}),   # critic caught it: never stored
    ("verified_only", True, {"same": 1.0, "related": 1.0, "unrelated": 0.0}),    # got past the critic: no help
    ("session_scope", True, {"same": 0.0, "related": 0.0, "unrelated": 0.0}),    # other users can't see it
])
async def test_contamination_under_each_defense(config, redis, setting, verified, expected):
    mem = FakeMemory()
    out = await run_memory_experiment(config, redis, "crispr gene therapy results", VICTIMS,
                                      DEFENSE_SETTINGS[setting], verified, writer_that_copies_background,
                                      ops=mem.ops(), run_key="t1")
    assert out["contamination"] == expected
    assert mem.cache == {} and mem.ltm == {}                     # everything planted was cleaned up
    assert config.memory_scope == "global"                       # shared config untouched


async def test_no_background_means_no_llm_call(config, redis):
    calls = []

    async def writer(topic, background):
        calls.append(topic)
        return ""

    mem = FakeMemory()
    out = await run_memory_experiment(config, redis, "crispr gene therapy results", [("unrelated", "ev sales trends")],
                                      DEFENSE_SETTINGS["none"], False, writer, ops=mem.ops(), run_key="t2")
    assert calls == [] and out["victims"][0]["path"] == "none"


async def test_cleanup_happens_even_if_a_victim_fails(config, redis):
    async def broken(topic, background):
        raise RuntimeError("LLM daily limit reached")

    mem = FakeMemory()
    with pytest.raises(RuntimeError):
        await run_memory_experiment(config, redis, "crispr gene therapy results", [("related", "gene editing ethics")],
                                    DEFENSE_SETTINGS["none"], False, broken, ops=mem.ops(), run_key="t3")
    assert mem.cache == {} and mem.ltm == {}
