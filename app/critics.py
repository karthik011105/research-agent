"""Step 3: a fact-checking critic panel with debate.

  claims    the report is split into single claims, each tied to the [n] sources it cites (code)
  passages  for each claim, the most relevant passages of its sources are picked by word overlap (code)
  critics   a fact-checker (gpt-oss-120b) and a skeptic (Qwen, another model family) give a verdict per claim,
            each quoting the source text it relies on
  judge     code: a verdict whose quote isn't really in the source is discarded; agreement is final
  debate    only for disagreements: each critic sees the other's verdict and quote and may revise (max N rounds);
            a remaining tie goes to the stricter verdict
  result    per claim keep / fix / remove, and exact fixes for the writer

CRITIC_MODE chooses "single" (the one YES/NO critic), "panel" (two critics, no debate) or "debate".
"""
import json
import logging
import random
import re
from dataclasses import dataclass, field
from app.config import Config

logger = logging.getLogger(__name__)

PROBLEM_VERDICTS = ("unsupported", "misattributed", "outdated", "exaggerated")
VERDICTS = ("supported",) + PROBLEM_VERDICTS
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have", "its", "their", "into",
    "than", "more", "less", "also", "which", "while", "about", "over", "under", "after", "before", "between", "per",
}
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
_ATTRIBUTION = re.compile(r"\b(said|says|according to|reported|reports|announced|stated|claims?|found|"
                          r"published|released|proposed|led by|developed by|introduced)\b", re.I)


@dataclass
class Claim:
    id: int
    text: str
    source_ids: list[int]
    kind: str                                   # number | date | attribution | general
    passages: list[str] = field(default_factory=list)


# ── claims ───────────────────────────────────────────────────────────────────

def claim_kind(text: str) -> str:
    plain = _CITATION.sub("", text)
    if re.search(rf"\b(19|20)\d\d\b|\b{_MONTHS}\b|\bQ[1-4]\b", plain, re.I):
        return "date"
    if re.search(r"\d", plain):
        return "number"
    if _ATTRIBUTION.search(plain):
        return "attribution"
    return "general"


def _sentences(body: str) -> list[str]:
    lines = [ln.strip(" -*•\t") for ln in body.splitlines()]
    out = []
    for line in lines:
        if not line or line.startswith("#") or len(line) < 25:
            continue
        # split after a sentence end, keeping a trailing citation like "...grew 20%. [3]" with its sentence
        out.extend(p.strip() for p in re.split(r"(?<=[.!?])(?:\s*\[\d+(?:\s*,\s*\d+)*\])?\s+(?=[A-Z])", line) if p.strip())
    return out


def split_claims(report: str, max_claims: int = 30) -> tuple[list[Claim], list[str]]:
    """(cited claims, uncited sentences that state numbers or dates). Code only, no LLM."""
    body = re.split(r"\n#+\s*Sources\s*\n", report)[0]
    claims, uncited = [], []
    for sentence in _sentences(body):
        ids = sorted({int(n) for g in _CITATION.findall(sentence) for n in re.split(r"\s*,\s*", g)})
        if ids:
            claims.append(Claim(id=len(claims) + 1, text=sentence, source_ids=ids, kind=claim_kind(sentence)))
        elif claim_kind(sentence) in ("number", "date"):
            uncited.append(sentence)
    return claims[:max_claims], uncited


# ── passages ─────────────────────────────────────────────────────────────────

def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9.%\-]*", _CITATION.sub("", text).lower())
    return {w.strip(".") for w in words if (len(w) > 2 or w[0].isdigit()) and w not in _STOP}


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text) - overlap, step)]


def best_passages(claim: Claim, sources: list[dict], per_source: int = 2, size: int = 380) -> list[str]:
    """The passages of the claim's cited sources that share the most terms with it (numbers count double)."""
    terms = _terms(claim.text)
    by_id = {s["id"]: s for s in sources}
    picked = []
    for sid in claim.source_ids:
        src = by_id.get(sid)
        if not src:
            continue
        scored = []
        for chunk in _chunks(src.get("content", ""), size, overlap=80):
            ct = _terms(chunk)
            score = sum(2 if t[0].isdigit() else 1 for t in terms & ct)
            scored.append((score, chunk))
        for score, chunk in sorted(scored, key=lambda x: -x[0])[:per_source]:
            if score > 0 or not picked:
                picked.append(f"[{sid}] {chunk}")
    return picked


# ── quotes ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\"'“”‘’`]", "", text).lower().split())


def quote_is_real(quote: str, sources: list[dict], source_ids: list[int], min_len: int = 12) -> bool:
    """True if the quote really appears (whitespace/case/quote-mark insensitive) in one of the cited sources."""
    q = _norm(quote or "").strip(" .…")
    if len(q) < min_len:
        return False
    q = q.split("...")[0].split("…")[0].strip() if ("..." in q or "…" in q) else q
    texts = [_norm(s.get("content", "")) for s in sources if s["id"] in source_ids]
    return any(q in t for t in texts)


# ── verdicts and the code judge ──────────────────────────────────────────────

def clean_verdict(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None
    try:
        claim_id = int(raw.get("id"))
    except (TypeError, ValueError):
        return None
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return None
    return {"id": claim_id, "verdict": verdict, "quote": str(raw.get("quote") or "")[:400],
            "fix": str(raw.get("fix") or "")[:300]}


def valid_vote(verdict: dict | None, claim: Claim, sources: list[dict]) -> bool:
    """A vote counts only if its evidence checks out: "supported" (and misattributed/outdated/exaggerated, which
    point at what the source really says) must quote real source text. "unsupported" may have no quote - the
    source is silent - but a quote it does give must be real."""
    if verdict is None:
        return False
    if verdict["verdict"] == "unsupported" and not verdict["quote"].strip():
        return True
    return quote_is_real(verdict["quote"], sources, claim.source_ids)


def stricter(a: dict, b: dict) -> dict:
    """Tie-break: a problem verdict beats "supported"."""
    return a if a["verdict"] != "supported" else b


def decide(claim: Claim, votes: list[dict | None], sources: list[dict], trust=None) -> dict:
    """Final verdict for one claim from the critics' votes, in CRITICS order (code, no LLM).
    A disagreement goes to the stricter verdict, or - once Step 4a has learned something - to the critic that is
    more trusted for this kind of claim."""
    named = [(role, v) for (role, _), v in zip(CRITICS, votes) if valid_vote(v, claim, sources)]
    valid = [v for _, v in named]
    if not valid:
        return {"verdict": "unverified", "decision": "fix", "fix": "No critic could verify this against its source: "
                "check it or remove it.", "agreed": False}
    if len(valid) == 1 or all((v["verdict"] == "supported") == (valid[0]["verdict"] == "supported") for v in valid):
        chosen, agreed = valid[0], len(valid) > 1
    else:
        agreed = False
        score = trust.weighted_problem_score(claim.kind, {r: v["verdict"] != "supported" for r, v in named}) \
            if trust is not None else 0.5
        if score == 0.5:
            chosen = stricter(valid[0], valid[1])
        else:
            chosen = next(v for v in valid if (v["verdict"] != "supported") == (score > 0.5))
    if chosen["verdict"] == "supported":
        return {"verdict": "supported", "decision": "keep", "fix": "", "agreed": agreed}
    decision = "fix" if chosen.get("fix") else "remove"
    return {"verdict": chosen["verdict"], "decision": decision, "fix": chosen.get("fix", ""), "agreed": agreed}


def disagreements(claims: list[Claim], a: dict, b: dict, sources: list[dict]) -> list[Claim]:
    """Claims where both critics cast a valid vote and one says supported while the other doesn't."""
    out = []
    for c in claims:
        va, vb = a.get(c.id), b.get(c.id)
        if valid_vote(va, c, sources) and valid_vote(vb, c, sources) and \
                (va["verdict"] == "supported") != (vb["verdict"] == "supported"):
            out.append(c)
    return out


def parse_verdicts(reply: str) -> dict[int, dict]:
    match = re.search(r"\{.*\}", reply, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}
    items = data.get("verdicts") if isinstance(data, dict) else None
    out = {}
    for raw in items if isinstance(items, list) else []:
        v = clean_verdict(raw)
        if v:
            out[v["id"]] = v
    return out


def feedback_for_writer(claims: list[Claim], finals: dict[int, dict], uncited: list[str]) -> str:
    lines = []
    for c in claims:
        f = finals.get(c.id)
        if not f or f["decision"] == "keep":
            continue
        action = "REMOVE" if f["decision"] == "remove" else "FIX"
        detail = f" - {f['fix']}" if f.get("fix") else ""
        lines.append(f"- {action} ({f['verdict']}): \"{c.text[:200]}\"{detail}")
    for sentence in uncited[:8]:
        lines.append(f"- FIX (uncited): \"{sentence[:200]}\" - add the source it came from, or remove it")
    return "\n".join(lines)


# ── the critics (LLM) ────────────────────────────────────────────────────────

CRITICS = (("fact_checker", "critic_fact"), ("skeptic", "critic_skeptic"))
_BATCH = 6          # claims per critic call: keeps the skeptic (Qwen) under Groq's 1,000 output tokens/minute

_ROLE = {
    "fact_checker": "Check whether each claim is stated by the cited source passages.",
    "skeptic": "Look for what is wrong with each claim: credited to the wrong source or person, outdated, "
               "exaggerated (stronger, bigger or more certain than the source), or merging facts from two sources.",
}


def _claims_block(claims: list[Claim]) -> str:
    parts = []
    for c in claims:
        passages = "\n".join(f"<source>{p}</source>" for p in c.passages) or "<source>(source not available)</source>"
        parts.append(f"CLAIM {c.id} (type: {c.kind}): {c.text}\n{passages}")
    return "\n\n".join(parts)


def critic_prompt(role: str, claims: list[Claim]) -> str:
    return (
        f"{_ROLE[role]}\n"
        f"For every claim give a verdict: \"supported\", \"unsupported\" (the passages don't say it), "
        f"\"misattributed\", \"outdated\" or \"exaggerated\". Copy into \"quote\" the exact words from a passage "
        f"that your verdict relies on (\"\" only for unsupported). For any verdict except supported, put a short "
        f"correction in \"fix\".\n"
        f'Reply with JSON only: {{"verdicts": [{{"id": 1, "verdict": "supported", "quote": "...", "fix": ""}}]}}\n\n'
        f"{_claims_block(claims)}"
    )


def debate_prompt(role: str, claim: Claim, own: dict, other: dict, other_role: str) -> str:
    return (
        f"{_ROLE[role]}\nYou and another reviewer ({other_role.replace('_', ' ')}) disagree about this claim. "
        f"Re-read the passages, weigh their argument and give your final verdict. Change your mind only if their "
        f"quote shows you were wrong.\n"
        f"Your verdict: {own['verdict']} - quote: \"{own['quote']}\" - fix: \"{own['fix']}\"\n"
        f"Their verdict: {other['verdict']} - quote: \"{other['quote']}\" - fix: \"{other['fix']}\"\n"
        f'Reply with JSON only: {{"verdicts": [{{"id": {claim.id}, "verdict": "...", "quote": "...", "fix": "..."}}]}}\n\n'
        f"{_claims_block([claim])}"
    )


@dataclass
class PanelResult:
    passed: bool
    feedback: str
    claims: list[dict]          # per claim: text, kind, final verdict/decision, each critic's vote, debated?
    llm_calls: int
    # Step 4a online signal: (claim kind, {critic: False}) for every vote discarded for a fabricated quote
    trust_events: list = field(default_factory=list)


async def _ask(call, config: Config, function: str, prompt: str) -> dict[int, dict]:
    try:
        return parse_verdicts(await call(config, function, prompt))
    except Exception as e:      # a failed critic counts as no vote; the other critic and the code judge still decide
        logger.warning(f"{function} failed: {e}")
        return {}


async def run_panel(config: Config, report: str, sources: list[dict], debate: bool, call=None,
                    trust=None, policy=None, rng=None, full_debate: bool = False) -> PanelResult:
    """Two critics vote on every cited claim; disagreements are debated if `debate`; code decides.

    trust   (Step 4a) settles disagreements by learned per-claim-type trust instead of "stricter wins"
    policy  (Step 4b) chooses 0-2 debate rounds per disputed claim instead of the fixed CRITIC_DEBATE_ROUNDS
    full_debate  debate every disputed claim for the full round budget and record the verdict after each round,
                 so a labelled evaluation can teach the policy what every choice would have decided"""
    if call is None:
        from app.agents import _tz_call as call
    claims, uncited = split_claims(report)
    for c in claims:
        c.passages = best_passages(c, sources)
    votes: dict[str, dict[int, dict]] = {}
    calls = 0
    for role, function in CRITICS:       # one after the other: free-tier tokens-per-minute limits are shared
        votes[role] = {}
        for i in range(0, len(claims), _BATCH):
            votes[role].update(await _ask(call, config, function, critic_prompt(role, claims[i:i + _BATCH])))
            calls += 1

    fc, sk = votes["fact_checker"], votes["skeptic"]
    trust_events = []
    for c in claims:                      # a vote that cites text not in the source is a mistake, whatever the truth
        bad = {role: False for role, v in (("fact_checker", fc.get(c.id)), ("skeptic", sk.get(c.id)))
               if v is not None and not valid_vote(v, c, sources)}
        if bad:
            trust_events.append((c.kind, bad))

    debated, trajectory = set(), {}
    if debate:
        disputed = disagreements(claims, fc, sk, sources)
        budget = {c.id: (config.critic_debate_rounds if (full_debate or policy is None)
                         else policy.choose(c.kind, rng or random.Random())) for c in disputed}
        for c in disputed:
            trajectory[c.id] = {0: decide(c, [fc.get(c.id), sk.get(c.id)], sources, trust)["verdict"]}
        for round_no in range(1, max(budget.values(), default=0) + 1):
            active = [c for c in disputed if budget[c.id] >= round_no and c.id not in _settled(c, fc, sk, sources)]
            for c in active:
                debated.add(c.id)
                new_fc = await _ask(call, config, "critic_fact", debate_prompt("fact_checker", c, fc[c.id], sk[c.id], "skeptic"))
                new_sk = await _ask(call, config, "critic_skeptic", debate_prompt("skeptic", c, sk[c.id], fc[c.id], "fact_checker"))
                calls += 2
                if c.id in new_fc:
                    fc[c.id] = new_fc[c.id]
                if c.id in new_sk:
                    sk[c.id] = new_sk[c.id]
            for c in disputed:            # a settled claim keeps its verdict for the remaining rounds
                trajectory[c.id][round_no] = decide(c, [fc.get(c.id), sk.get(c.id)], sources, trust)["verdict"]

    finals = {c.id: decide(c, [fc.get(c.id), sk.get(c.id)], sources, trust) for c in claims}
    details = [{"id": c.id, "text": c.text, "kind": c.kind, "sources": c.source_ids, **finals[c.id],
                "debated": c.id in debated, "trajectory": trajectory.get(c.id),
                "fact_checker": fc.get(c.id), "skeptic": sk.get(c.id)} for c in claims]
    feedback = feedback_for_writer(claims, finals, uncited)
    return PanelResult(passed=not feedback, feedback=feedback, claims=details, llm_calls=calls,
                       trust_events=trust_events)


def _settled(claim: Claim, fc: dict, sk: dict, sources: list[dict]) -> set:
    """{claim.id} if the critics now agree on this claim, else an empty set."""
    a, b = fc.get(claim.id), sk.get(claim.id)
    if valid_vote(a, claim, sources) and valid_vote(b, claim, sources) and \
            (a["verdict"] == "supported") == (b["verdict"] == "supported"):
        return {claim.id}
    return set()
