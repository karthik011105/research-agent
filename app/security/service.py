"""Step 5 experiments as background runs on the deployed app (results in bench_runs, like the benchmark).

  attack runs        one poisoned page per run (5a), traced stage by stage (5b), under a chosen critic mode and
                     with the approved self-healing rules applied (5d)
  memory runs        memory-poisoning contamination under one defense setting (5c)
  rule proposals     an LLM rule from an attack page, tested on held-out phrasings and real benign pages (5d)
"""
import copy
import time
from datetime import datetime
from app.bench import BenchStore, snapshot_search
from app.config import Config
from app.security.attacks import ATTACKS_BY_ID, STYLES, poisoned_page
from app.security.memory_poisoning import DEFENSE_SETTINGS, run_memory_experiment
from app.security.propagation import run_attack
from app.security.rules import propose_rule


async def _finish(store: BenchStore, run: dict, work) -> dict:
    started = time.monotonic()
    try:
        metrics, report = await work()
        run.update(status="done", metrics={**metrics, "latency_s": round(time.monotonic() - started, 1)}, report=report)
    except Exception as e:
        run.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
    run["finished_at"] = datetime.utcnow()
    await store.save_run(run)
    return run


async def execute_attack_run(config: Config, redis, store: BenchStore, run: dict, attack_id: str,
                             critic_mode: str, rulebook=None) -> dict:
    async def work():
        result = await run_attack(config, redis, snapshot_search(store, run["topic_id"]), run["topic"], attack_id,
                                  run["id"][:8], critic_mode, rulebook=rulebook)
        final = result.pop("final_report", "")
        return {**result, "rules_active": len(rulebook.patterns) if rulebook else 0}, final
    return await _finish(store, run, work)


async def execute_memory_run(config: Config, redis, store: BenchStore, run: dict, plant_topic: str,
                             victims: list[tuple[str, str]], setting: str, poison_verified: bool, ops=None) -> dict:
    from app.agents import WriterAgent

    async def write_with_background(topic: str, background: str) -> str:
        # the writer with nothing but the remembered background: does memory content reach a new user's report?
        cfg = copy.copy(config)
        return await WriterAgent(cfg).run(topic, "(no new findings were retrieved for this request)", False,
                                          background, "")

    async def work():
        out = await run_memory_experiment(config, redis, plant_topic, victims, DEFENSE_SETTINGS[setting],
                                          poison_verified, write_with_background, ops=ops, run_key=run["id"][:8])
        return {k: v for k, v in out.items() if k != "victims"} | {"victims": out["victims"]}, None
    return await _finish(store, run, work)


async def propose_rule_for_attack(config: Config, store, rule_store, attack_id: str, call=None) -> dict:
    """Proposes a rule from style 0 of the attack and tests it on the held-out styles and on real saved pages."""
    attack = ATTACKS_BY_ID[attack_id]
    seed_page = poisoned_page(attack, "a recent technology topic", "seed", style=0)["content"]
    held_out = [poisoned_page(attack, f"topic {i}", f"held-{i}", style=s)["content"]
                for i in range(4) for s in range(1, STYLES)]
    benign = [s["content"] for s in await store.all_evidence(limit=300) if s.get("provider") != "redteam"]
    rule = await propose_rule(config, seed_page, held_out, benign, call=call)
    rule["attack_id"] = attack_id
    await rule_store.save(rule)
    return rule
