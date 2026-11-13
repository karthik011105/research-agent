"""Step 5d: self-healing guardrails.

When an attack gets through (Step 5b says "got_through"), an LLM proposes a filter rule - a regular expression for
the injection *technique* (e.g. "ignore your previous instructions"), not the attack's one-off wording. Code then
tests it before anyone can use it:

  recall           it must match the attack's OTHER phrasings (held out from the proposal)        >= MIN_RECALL
  false positives  it must match almost no real, harmless web pages (saved benchmark sources)    <= MAX_FALSE_POSITIVE
  safety           compiles, not too short or too broad, no nested quantifiers (slow regexes)

A rule that passes is only *proposed*: the owner approves it (POST /security/rules/{id}/approve). Approved rules
remove matching sentences from web pages before any LLM reads them (sanitize_sources, used in the merge step).
"""
import json
import re
import uuid
from datetime import datetime

MIN_RECALL = 0.8
MAX_FALSE_POSITIVE = 0.02
# A repeated group that itself repeats without bound, e.g. (a+)+ or (\w*)* or (\w+){2,}: classic catastrophic
# backtracking. Bounded outer repetition such as (\w+\s+){0,3} is fine.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)(?:[+*]|\{\d+,\})")
_SENTENCE = re.compile(r"[^.!?]*(?:[.!?]+|$)")


def check_pattern(pattern: str) -> str | None:
    """None if the pattern is acceptable, else the reason it is rejected."""
    if not isinstance(pattern, str) or not 6 <= len(pattern) <= 200:
        return "pattern must be 6-200 characters"
    if _NESTED_QUANTIFIER.search(pattern):
        return "nested quantifiers can make matching extremely slow"
    try:
        compiled = re.compile(pattern, re.I)
    except re.error as e:
        return f"invalid regex: {e}"
    if compiled.search("") or compiled.search("the and of to in a is"):
        return "pattern matches empty or ordinary text"
    return None


def evaluate_rule(pattern: str, attack_texts: list[str], benign_texts: list[str]) -> dict:
    compiled = re.compile(pattern, re.I)
    hits = sum(bool(compiled.search(t)) for t in attack_texts)
    fps = sum(bool(compiled.search(t)) for t in benign_texts)
    recall = hits / len(attack_texts) if attack_texts else 0.0
    fp_rate = fps / len(benign_texts) if benign_texts else 0.0
    return {"recall": round(recall, 3), "false_positive_rate": round(fp_rate, 3),
            "attack_texts": len(attack_texts), "benign_texts": len(benign_texts),
            "passed": recall >= MIN_RECALL and fp_rate <= MAX_FALSE_POSITIVE and len(benign_texts) >= 10}


def proposal_prompt(page_text: str) -> str:
    return (
        "A web page below tricked an AI research assistant. Write ONE case-insensitive regular expression that "
        "detects the manipulation TECHNIQUE it uses (for example an instruction aimed at AI systems, fake system "
        "markup, or a request to reveal instructions) so that other pages using the same technique with different "
        "wording are also caught. Do not match the specific facts, names or topic of this page, and do not match "
        "normal article text. Keep it under 150 characters and avoid nested quantifiers.\n"
        'Reply with JSON only: {"pattern": "...", "description": "<one sentence>"}\n\n'
        f"<source>{page_text[:2000]}</source>"
    )


async def propose_rule(config, page_text: str, attack_texts: list[str], benign_texts: list[str], call=None) -> dict:
    """Asks the LLM for a rule and tests it. Returns a rule record (status "proposed" or "rejected")."""
    if call is None:
        from app.agents import _tz_call as call
    from app.agents import parse_json_object
    reply = parse_json_object(await call(config, "security_rule", proposal_prompt(page_text)))
    pattern = str(reply.get("pattern") or "")
    problem = check_pattern(pattern)
    result = {"passed": False} if problem else evaluate_rule(pattern, attack_texts, benign_texts)
    return {
        "id": uuid.uuid4().hex[:12], "pattern": pattern, "description": str(reply.get("description", ""))[:300],
        "status": "proposed" if result["passed"] else "rejected", "problem": problem, "test": result,
        "created_at": datetime.utcnow().isoformat(),
    }


def sanitize_text(text: str, patterns: list[re.Pattern]) -> tuple[str, int]:
    """Drops every sentence that matches an approved rule. Returns (clean text, sentences removed)."""
    kept, removed = [], 0
    for sentence in _SENTENCE.findall(text):
        if not sentence.strip():
            continue
        if any(p.search(sentence) for p in patterns):
            removed += 1
        else:
            kept.append(sentence)
    return "".join(kept).strip(), removed


def sanitize_sources(sources: list[dict], patterns: list[re.Pattern]) -> list[dict]:
    if not patterns:
        return sources
    out = []
    for s in sources:
        clean, removed = sanitize_text(s.get("content", ""), patterns)
        out.append({**s, "content": clean, "sanitized": removed} if removed else s)
    return out


class RuleBook:
    """Approved rules, compiled, shared by the pipeline; reloaded after every approval."""

    def __init__(self, store=None):
        self.store = store
        self.patterns: list[re.Pattern] = []

    async def reload(self) -> None:
        if self.store is not None:
            self.patterns = [re.compile(r["pattern"], re.I) for r in await self.store.list_rules() if r["status"] == "approved"]


class PostgresRuleStore:
    def __init__(self, pool):
        self.pool = pool

    async def migrate(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_rules (
                    id TEXT PRIMARY KEY, rule JSONB NOT NULL, status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW())""")

    async def save(self, rule: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO security_rules (id, rule, status) VALUES ($1, $2::jsonb, $3) "
                "ON CONFLICT (id) DO UPDATE SET rule=EXCLUDED.rule, status=EXCLUDED.status",
                rule["id"], json.dumps(rule), rule["status"])

    async def list_rules(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT rule, status FROM security_rules ORDER BY created_at")
        return [{**json.loads(r["rule"]), "status": r["status"]} for r in rows]

    async def get(self, rule_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT rule, status FROM security_rules WHERE id=$1", rule_id)
        return {**json.loads(row["rule"]), "status": row["status"]} if row else None
