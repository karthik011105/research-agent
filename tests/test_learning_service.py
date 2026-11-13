import pytest
from app import agents, metering
from app.agents import OrchestratorAgent, initial_state
from app.learning import LearningService, outcome_parts
from tests.conftest import make_source


class MemoryLearningStore:
    def __init__(self):
        self.state, self.writer = {}, {}

    async def get(self, key):
        return self.state.get(key)

    async def put(self, key, state):
        self.state[key] = state

    async def record_writer(self, job_id, variant, reward, parts):
        self.writer[job_id] = {"variant": variant, "reward": reward, "parts": parts}

    async def writer_job(self, job_id):
        return self.writer.get(job_id)

    async def writer_history(self, limit=500):
        return [{"variant": v["variant"], "reward": v["reward"]} for v in self.writer.values()]


def test_outcome_parts():
    sources = [{**make_source(i), "id": i} for i in (1, 2)]
    report = "## Summary\nA [1]. B [2]. C [9].\n\n## Sources\n[1] x"
    panel = {"claim_verdicts": [{"decision": "keep"}, {"decision": "keep"}, {"decision": "fix"}, {"decision": "keep"}],
             "verified": False, "iterations": 2}
    p = outcome_parts(panel, report, sources, 95.4)
    assert p == {"claim_pass_rate": 0.75, "citation_validity": 0.667, "first_draft_ok": 0.0, "latency_s": 95.4}
    single = outcome_parts({"verified": True, "iterations": 1}, report, sources, 30)
    assert single["claim_pass_rate"] == 1.0 and single["first_draft_ok"] == 1.0


async def test_service_rewards_feedback_and_persists():
    store = MemoryLearningStore()
    svc = LearningService(store, enabled=True, seed=1)
    await svc.load()
    reward = await svc.record_report("job-1", "cite_numbers", {"claim_pass_rate": 1.0, "citation_validity": 1.0})
    assert reward == 1.0 and store.state["writer_bandit"]["arms"]["cite_numbers"]["n"] == 1
    out = await svc.apply_feedback("job-1", -1)
    assert out["variant"] == "cite_numbers" and store.writer["job-1"]["parts"]["user_rating"] == -1
    assert store.state["writer_bandit"]["arms"]["cite_numbers"]["n"] == 2
    assert await svc.apply_feedback("unknown-job", 1) is None

    svc2 = LearningService(store, enabled=True)
    await svc2.load()                                   # another worker / a restart sees the same state
    assert svc2.writer.arms["cite_numbers"]["n"] == 2
    stats = await svc2.stats()
    assert set(stats["writer"]["win_probability"]) == {"primary", "writer_20b", "cite_numbers"}
    assert stats["debate_policy"]["number"]["best_rounds"] in (0, 1, 2)


async def test_disabled_service_stays_rule_based():
    svc = LearningService(MemoryLearningStore(), enabled=False)
    assert svc.choose_writer() is None and svc.critic_learners() == (None, None)


async def test_trust_events_from_the_panel_are_applied():
    store = MemoryLearningStore()
    svc = LearningService(store, enabled=True)
    await svc.apply_trust_events([("number", {"skeptic": False}), ("number", {"skeptic": False})])
    assert svc.trust.weights["number"]["skeptic"] < 0.5 and "critic_trust" in store.state


class WriterRecorder:
    def __init__(self):
        self.variants = []

    async def __call__(self, config, function_name, message, variant=None):
        if function_name == "report_write":
            self.variants.append(variant)
        return "## Executive Summary\nFact [1]."


@pytest.fixture
def svc():
    s = LearningService(MemoryLearningStore(), enabled=True, seed=4)
    s.writer.arms["cite_numbers"] = {"a": 500.0, "b": 1.0}       # make the choice predictable
    s.explore = s.writer.explore = 0.0
    return s


async def test_write_node_uses_and_keeps_the_bandits_variant(config, redis, monkeypatch, svc):
    rec = WriterRecorder()
    monkeypatch.setattr(agents, "_tz_call", rec)
    orch = OrchestratorAgent(config, redis, learning=svc)
    state = initial_state("topic", "s", [], "")
    state.update(facts="- f [1]", live_sources=True, sources=[{**make_source(1), "id": 1}])
    first = await orch.write_node(state)
    assert first["writer_variant"] == "cite_numbers"
    state.update(first)
    await orch.write_node(state)                                   # rewrite after critic feedback
    assert rec.variants == ["cite_numbers", "cite_numbers"]


async def test_benchmark_runs_never_use_the_bandit(config, redis, monkeypatch, svc):
    rec = WriterRecorder()
    monkeypatch.setattr(agents, "_tz_call", rec)
    orch = OrchestratorAgent(config, redis, learning=svc)
    state = initial_state("topic", "s", [], "")
    state.update(facts="- f [1]", live_sources=True, sources=[{**make_source(1), "id": 1}])
    metering.start()
    try:
        out = await orch.write_node(state)
    finally:
        metering._counters.set(None)
    assert out["writer_variant"] == "" and rec.variants == [None]
