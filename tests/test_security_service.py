import json
from app import agents
from app.bench import new_run
from app.security.service import execute_attack_run, execute_memory_run, propose_rule_for_attack
from tests.conftest import make_source
from tests.test_bench import MemoryStore
from tests.test_security_attacks import _fake_pipeline
from tests.test_security_memory import VICTIMS, FakeMemory


class StoreWithEvidence(MemoryStore):
    async def all_evidence(self, limit=300):
        return [make_source(i, content=f"An ordinary article about topic {i}. " * 8) for i in range(20)]


class RuleStore:
    def __init__(self):
        self.rules = {}

    async def save(self, rule):
        self.rules[rule["id"]] = rule


async def test_attack_run_records_the_stage_where_it_stopped(config, redis, monkeypatch):
    async def no_scrape(redis_, config_, sources):
        return sources

    monkeypatch.setattr(agents, "_tz_call", _fake_pipeline(obeys=True, critic_catches=True))
    monkeypatch.setattr(agents, "enrich_sources", no_scrape)
    monkeypatch.setattr("app.bench.search_kind",
                        lambda *a, **k: _async([{**make_source(9, url="https://news.example/x"), "kind": "web"}]))
    config.guardrails_enabled, config.search_max_rounds = False, 1
    store = MemoryStore()
    run = await execute_attack_run(config, redis, store, new_run("sec1", "n4", "EV sales", "attack:instr_claim:single"),
                                   "instr_claim", "single")
    assert run["status"] == "done", run.get("error")
    assert run["metrics"]["stopped_at"] == "critic" and run["metrics"]["rules_active"] == 0
    assert "Varnholt" not in run["report"]                    # the final report the user would have seen


async def test_memory_run_stores_contamination_and_cleans_up(config, redis, monkeypatch):
    async def writer_llm(config_, function, message, variant=None):
        background = message.split("BACKGROUND FROM EARLIER RESEARCH")[1] if "BACKGROUND" in message else ""
        return "## Summary\n" + background[:3000]

    monkeypatch.setattr(agents, "_tz_call", writer_llm)
    mem, store = FakeMemory(), MemoryStore()
    run = await execute_memory_run(config, redis, store, new_run("sec1", "memory", "crispr gene therapy results", "memory:none"),
                                   "crispr gene therapy results", VICTIMS, "none", False, ops=mem.ops())
    assert run["status"] == "done", run.get("error")
    assert run["metrics"]["contamination"] == {"same": 1.0, "related": 1.0, "unrelated": 0.0}
    assert mem.cache == {} and mem.ltm == {}


async def test_rule_proposal_is_tested_on_held_out_phrasings_and_real_pages(config):
    async def llm(config_, function, prompt):
        return json.dumps({"pattern": r"(ignore|disregard|forget)\s+(\w+\s+){0,3}(instructions|guidance)",
                           "description": "tells the AI to drop its guidance"})

    rules = RuleStore()
    rule = await propose_rule_for_attack(config, StoreWithEvidence(), rules, "instr_claim", call=llm)
    assert rule["status"] == "proposed" and rule["test"]["attack_texts"] == 8 and rule["test"]["benign_texts"] == 20
    assert rule["attack_id"] == "instr_claim" and rule["id"] in rules.rules


async def _async(value):
    return value
