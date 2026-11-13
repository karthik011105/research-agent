"""Step 4: learning from scores. Nothing here retrains a model: the system learns *decisions* from outcomes.

  CriticTrust      4a  how much to believe each critic per claim type      (multiplicative weights / Hedge)
  DebatePolicy     4b  how many debate rounds a disputed claim deserves     (contextual Thompson sampling)
  WriterBandit     4c  which writer variant to use for a report            (Thompson sampling, Beta posteriors)

State is small JSON kept in Postgres (learning_state) through a LearningStore, so every worker shares it and it
survives restarts. All three keep exploring a little and can be switched off (LEARNING_ENABLED=false).
"""
import json
import math
import random
from typing import Protocol

KINDS = ("number", "date", "attribution", "general")
CRITICS = ("fact_checker", "skeptic")
DEBATE_ARMS = (0, 1, 2)                       # debate rounds for a disputed claim
WRITER_VARIANTS = ("primary", "writer_20b", "cite_numbers")


class LearningStore(Protocol):
    async def get(self, key: str) -> dict | None: ...
    async def put(self, key: str, state: dict) -> None: ...


# ── 4a: critic trust (Hedge / multiplicative weights) ───────────────────────

class CriticTrust:
    """One weight per critic per claim type, starting equal. After a verdict whose truth is known, the critic that
    was wrong is multiplied by exp(-eta); weights are renormalised per claim type. Weights never reach zero, so a
    critic that improves can win trust back."""

    def __init__(self, state: dict | None = None, eta: float = 0.3, floor: float = 0.05):
        self.eta, self.floor = eta, floor
        self.weights = {k: {c: 1 / len(CRITICS) for c in CRITICS} for k in KINDS}
        self.updates = {k: 0 for k in KINDS}
        if state:
            for k, ws in state.get("weights", {}).items():
                self.weights.setdefault(k, {}).update(ws)
            self.updates.update(state.get("updates", {}))

    def update(self, kind: str, correct: dict[str, bool]) -> None:
        kind = kind if kind in self.weights else "general"
        ws = self.weights[kind]
        for critic, ok in correct.items():
            if critic in ws and not ok:
                ws[critic] *= math.exp(-self.eta)
        total = sum(ws.values())
        for critic in ws:
            ws[critic] = max(self.floor, ws[critic] / total)
        total = sum(ws.values())
        for critic in ws:
            ws[critic] /= total
        self.updates[kind] += 1

    def weighted_problem_score(self, kind: str, votes: dict[str, bool]) -> float:
        """votes: critic -> True if it found a problem. Returns the trust-weighted share saying "problem"."""
        ws = self.weights.get(kind, self.weights["general"])
        voted = {c: v for c, v in votes.items() if c in ws}
        total = sum(ws[c] for c in voted)
        return sum(ws[c] for c, v in voted.items() if v) / total if total else 0.5

    def state(self) -> dict:
        return {"weights": self.weights, "updates": self.updates}


# ── Beta posteriors for Thompson sampling ───────────────────────────────────

def beta_update(arm: dict, reward: float) -> None:
    """Fractional Bernoulli update: a reward in [0, 1] adds r to alpha and 1-r to beta."""
    r = min(1.0, max(0.0, reward))
    arm["a"] = arm.get("a", 1.0) + r
    arm["b"] = arm.get("b", 1.0) + (1 - r)
    arm["n"] = arm.get("n", 0) + 1


def beta_mean(arm: dict) -> float:
    a, b = arm.get("a", 1.0), arm.get("b", 1.0)
    return a / (a + b)


def thompson_choice(arms: dict[str, dict], rng: random.Random, explore: float = 0.1) -> str:
    """Sample each arm's Beta and take the best; with probability `explore` pick uniformly (exploration floor)."""
    names = list(arms)
    if rng.random() < explore:
        return rng.choice(names)
    return max(names, key=lambda n: rng.betavariate(arms[n].get("a", 1.0), arms[n].get("b", 1.0)))


def win_probabilities(arms: dict[str, dict], samples: int = 4000, seed: int = 0) -> dict[str, float]:
    """Monte-Carlo probability that each arm has the highest true mean (for the dashboard)."""
    rng = random.Random(seed)
    wins = dict.fromkeys(arms, 0)
    for _ in range(samples):
        best = max(arms, key=lambda n: rng.betavariate(arms[n].get("a", 1.0), arms[n].get("b", 1.0)))
        wins[best] += 1
    return {n: round(w / samples, 3) for n, w in wins.items()}


# ── 4b: learned debate policy (contextual bandit) ───────────────────────────

class DebatePolicy:
    """Context = claim type of a claim the critics disagree on; arms = 0, 1 or 2 debate rounds.
    Reward = 1 if the final verdict was right, minus `call_cost` per debate round (both critics speak each round).
    A 2-round debate also shows what 0 and 1 rounds would have decided, so one labelled example updates every arm
    (full information), which makes learning fast on a small free-tier budget."""

    def __init__(self, state: dict | None = None, call_cost: float = 0.15, explore: float = 0.1):
        self.call_cost, self.explore = call_cost, explore
        self.arms = {k: {str(r): {"a": 1.0, "b": 1.0, "n": 0} for r in DEBATE_ARMS} for k in KINDS}
        if state:
            for k, arms in state.get("arms", {}).items():
                self.arms.setdefault(k, {}).update(arms)

    def reward(self, correct: bool, rounds: int) -> float:
        return max(0.0, (1.0 if correct else 0.0) - self.call_cost * rounds)

    def learn(self, kind: str, correct_after_rounds: dict[int, bool]) -> None:
        """correct_after_rounds: rounds -> was the verdict right if we had stopped after that many rounds."""
        arms = self.arms.get(kind, self.arms["general"])
        for rounds, ok in correct_after_rounds.items():
            beta_update(arms[str(rounds)], self.reward(ok, rounds))

    def choose(self, kind: str, rng: random.Random) -> int:
        return int(thompson_choice(self.arms.get(kind, self.arms["general"]), rng, self.explore))

    def best(self, kind: str) -> int:
        arms = self.arms.get(kind, self.arms["general"])
        return int(max(arms, key=lambda r: beta_mean(arms[r])))

    def state(self) -> dict:
        return {"arms": self.arms}


# ── 4c: writer variant selection ────────────────────────────────────────────

def report_reward(parts: dict) -> float:
    """Reward in [0, 1] for one report from mostly objective signals; missing parts are skipped (re-weighted).
      claim_pass_rate      share of claims the critic panel kept (or 1/0 for the single critic's YES/NO)
      citation_validity    share of [n] markers pointing at a real source
      first_draft_ok       the critic accepted the first draft (no rewrite needed)
      user_rating          +1 / -1 from the 👍/👎 buttons, mapped to 1 / 0
      latency_s            small penalty above 120 s"""
    weights = {"claim_pass_rate": 0.4, "citation_validity": 0.25, "first_draft_ok": 0.15, "user_rating": 0.2}
    total, used = 0.0, 0.0
    for key, w in weights.items():
        value = parts.get(key)
        if value is None:
            continue
        if key == "user_rating":
            value = 1.0 if value > 0 else 0.0
        total += w * float(value)
        used += w
    score = total / used if used else 0.5
    latency = parts.get("latency_s")
    if latency and latency > 120:
        score -= min(0.1, (latency - 120) / 1800)
    return round(min(1.0, max(0.0, score)), 4)


class WriterBandit:
    def __init__(self, state: dict | None = None, explore: float = 0.1):
        self.explore = explore
        self.arms = {v: {"a": 1.0, "b": 1.0, "n": 0} for v in WRITER_VARIANTS}
        if state:
            self.arms.update({k: v for k, v in state.get("arms", {}).items() if k in self.arms})

    def choose(self, rng: random.Random) -> str:
        return thompson_choice(self.arms, rng, self.explore)

    def learn(self, variant: str, reward: float) -> None:
        if variant in self.arms:
            beta_update(self.arms[variant], reward)

    def summary(self) -> dict:
        return {"arms": {v: {**a, "mean": round(beta_mean(a), 3)} for v, a in self.arms.items()},
                "win_probability": win_probabilities(self.arms)}

    def state(self) -> dict:
        return {"arms": self.arms}


# ── Postgres storage ────────────────────────────────────────────────────────

class PostgresLearningStore:
    def __init__(self, pool):
        self.pool = pool

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_state (
                    key TEXT PRIMARY KEY, state JSONB NOT NULL, updated_at TIMESTAMP NOT NULL DEFAULT NOW())""")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS writer_rewards (
                    job_id TEXT PRIMARY KEY, variant TEXT NOT NULL, reward REAL, parts JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW())""")

    async def get(self, key):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT state FROM learning_state WHERE key=$1", key)
        return json.loads(row["state"]) if row else None

    async def put(self, key, state):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO learning_state (key, state, updated_at) VALUES ($1, $2::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET state=EXCLUDED.state, updated_at=NOW()", key, json.dumps(state))

    async def record_writer(self, job_id: str, variant: str, reward: float | None, parts: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO writer_rewards (job_id, variant, reward, parts) VALUES ($1,$2,$3,$4::jsonb) "
                "ON CONFLICT (job_id) DO UPDATE SET reward=EXCLUDED.reward, parts=EXCLUDED.parts",
                job_id, variant, reward, json.dumps(parts))

    async def writer_job(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT variant, reward, parts FROM writer_rewards WHERE job_id=$1", job_id)
        return {"variant": row["variant"], "reward": row["reward"], "parts": json.loads(row["parts"])} if row else None

    async def writer_history(self, limit: int = 500) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT variant, reward, created_at FROM writer_rewards WHERE reward IS NOT NULL "
                "ORDER BY created_at DESC LIMIT $1", limit)
        return [{"variant": r["variant"], "reward": r["reward"], "at": r["created_at"].isoformat()} for r in rows][::-1]


# ── the service the app uses ────────────────────────────────────────────────

class LearningService:
    """Holds the three learners, persists them, and turns job outcomes into rewards.

    `enabled` (LEARNING_ENABLED) controls whether learned behaviour is *used*: the writer bandit choosing variants
    and the critic panel using learned trust and debate rounds. When it is off everything stays rule-based (the
    baseline Step 4 must beat), but labelled critic evaluations still train the critic learners."""

    def __init__(self, store: LearningStore, enabled: bool = False, explore: float = 0.1, seed: int | None = None):
        import asyncio
        self.store, self.enabled, self.explore = store, enabled, explore
        self.rng = random.Random(seed)
        self.lock = asyncio.Lock()
        self.trust, self.policy, self.writer = CriticTrust(), DebatePolicy(explore=explore), WriterBandit(explore=explore)

    async def load(self) -> None:
        self.trust = CriticTrust(await self.store.get("critic_trust"))
        self.policy = DebatePolicy(await self.store.get("debate_policy"), explore=self.explore)
        self.writer = WriterBandit(await self.store.get("writer_bandit"), explore=self.explore)

    async def save(self) -> None:
        await self.store.put("critic_trust", self.trust.state())
        await self.store.put("debate_policy", self.policy.state())
        await self.store.put("writer_bandit", self.writer.state())

    # -- used by the pipeline --
    def choose_writer(self) -> str | None:
        return self.writer.choose(self.rng) if self.enabled else None

    def critic_learners(self) -> tuple:
        """(trust, policy) for the critic panel, or (None, None) to stay rule-based."""
        return (self.trust, self.policy) if self.enabled else (None, None)

    async def apply_trust_events(self, events: list) -> None:
        if not events:
            return
        async with self.lock:
            for kind, correct in events:
                self.trust.update(kind, correct)
            await self.store.put("critic_trust", self.trust.state())

    # -- rewards for the writer bandit --
    async def record_report(self, job_id: str, variant: str, parts: dict) -> float:
        reward = report_reward(parts)
        async with self.lock:
            self.writer.learn(variant, reward)
            await self.store.put("writer_bandit", self.writer.state())
        if hasattr(self.store, "record_writer"):
            await self.store.record_writer(job_id, variant, reward, parts)
        return reward

    async def apply_feedback(self, job_id: str, rating: int) -> dict | None:
        """A 👍/👎 is one more observation for the variant that wrote the report."""
        job = await self.store.writer_job(job_id) if hasattr(self.store, "writer_job") else None
        if job is None:
            return None
        parts = {**job["parts"], "user_rating": rating}
        async with self.lock:
            self.writer.learn(job["variant"], 1.0 if rating > 0 else 0.0)
            await self.store.put("writer_bandit", self.writer.state())
        reward = report_reward(parts)
        await self.store.record_writer(job_id, job["variant"], reward, parts)
        return {"variant": job["variant"], "reward": reward}

    async def stats(self) -> dict:
        history = await self.store.writer_history() if hasattr(self.store, "writer_history") else []
        return {
            "enabled": self.enabled,
            "writer": {**self.writer.summary(), "history": history},
            "critic_trust": self.trust.state(),
            "debate_policy": {k: {"best_rounds": self.policy.best(k),
                                  "arms": {r: {"mean": round(beta_mean(a), 3), "n": a.get("n", 0)}
                                           for r, a in arms.items()}}
                              for k, arms in self.policy.arms.items()},
        }


def outcome_parts(final_state: dict, report: str, sources: list[dict], latency_s: float) -> dict:
    """Reward signals for the writer bandit from a finished pipeline run (objective ones only; 👍/👎 comes later)."""
    import re
    from app.agents import cited_ids
    verdicts = final_state.get("claim_verdicts") or []
    if verdicts:
        claim_pass_rate = sum(v.get("decision") == "keep" for v in verdicts) / len(verdicts)
    else:
        claim_pass_rate = 1.0 if final_state.get("verified") else 0.0
    body = re.split(r"\n#+\s*Sources\s*\n", report)[0]
    cited = cited_ids(body)
    valid = {s.get("id") for s in sources}
    return {
        "claim_pass_rate": round(claim_pass_rate, 3),
        "citation_validity": round(len(cited & valid) / len(cited), 3) if cited and sources else None,
        "first_draft_ok": 1.0 if final_state.get("iterations", 0) <= 1 and final_state.get("verified") else 0.0,
        "latency_s": round(latency_s, 1),
    }
