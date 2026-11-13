"""Step 3 measurement: labelled claims for free (error injection), and scoring of critic setups.

True claims are sentences taken from the sources themselves (so they are supported by construction). False
claims are copies broken in a known way:
  number       a figure changed (x1.5 or +3)
  date         a year shifted
  attribution  a named organisation/person swapped for one from another source
  merge        a statement joined with a clause from another source
Each setup ("single", "panel", "debate") checks the same claims; we count errors caught and false alarms.
"""
import random
import re
from app.config import Config

_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?=\s*(%|percent|million|billion|thousand|units|people|users|\b))")
_NAME = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2})\b")
_GENERIC = {"The", "This", "These", "That", "However", "According", "In", "On", "At", "For", "And", "But", "It", "A",
            "An", "As", "By", "Its", "Their", "Our", "We", "They", "He", "She", "There", "When", "While", "With"}


def source_sentences(source: dict, min_len: int = 50, max_len: int = 220) -> list[str]:
    text = " ".join(source.get("content", "").split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p for p in parts if min_len <= len(p) <= max_len and re.search(r"\d|[A-Z][a-z]+\s+[A-Z]", p[1:])]


def perturb_number(sentence: str) -> str | None:
    for m in _NUMBER.finditer(sentence):
        value = m.group(1)
        if _YEAR.fullmatch(value):
            continue
        num = float(value)
        new = num * 1.5 if num >= 4 else num + 3
        new_text = f"{new:.1f}" if "." in value else str(int(round(new)))
        return sentence[:m.start(1)] + new_text + sentence[m.end(1):]
    return None


def perturb_date(sentence: str) -> str | None:
    m = _YEAR.search(sentence)
    if not m:
        return None
    year = int(m.group(1))
    return sentence[:m.start()] + str(year - 2 if year >= 2026 else year + 2) + sentence[m.end():]


def names(sentence: str) -> list[str]:
    """Capitalised names, skipping a single capitalised word that only starts the sentence ("Analysts at ...")."""
    out = []
    for m in _NAME.finditer(sentence):
        n = m.group(1)
        if n.split()[0] in _GENERIC or _YEAR.fullmatch(n) or (m.start() == 0 and " " not in n):
            continue
        out.append(n)
    return out


def perturb_attribution(sentence: str, other_names: list[str]) -> str | None:
    mine = names(sentence)
    candidates = [n for n in other_names if n not in mine and n not in sentence]
    if not mine or not candidates:
        return None
    return sentence.replace(mine[0], candidates[0], 1)


def perturb_merge(sentence: str, other_sentence: str) -> str:
    clause = other_sentence.rstrip(".").split(",")[0]
    clause = clause[0].lower() + clause[1:] if clause else clause
    return f"{sentence.rstrip('.')}, and {clause}."


def build_labelled_claims(sources: list[dict], n: int = 16, seed: int = 7) -> list[dict]:
    """About half true claims (verbatim from a source), half injected errors, each citing the source it came from.
    Deterministic for a given seed and source list."""
    rng = random.Random(seed)
    pool = [(s, sent) for s in sources for sent in source_sentences(s)]
    rng.shuffle(pool)
    all_names = [nm for s in sources for sent in source_sentences(s) for nm in names(sent)]
    claims, kinds = [], ["number", "date", "attribution", "merge"]
    for i, (src, sent) in enumerate(pool):
        if len(claims) >= n:
            break
        if i % 2 == 0:
            claims.append({"text": sent, "source_id": src["id"], "label": "true", "error": None})
            continue
        others = [x for x in pool if x[0]["id"] != src["id"]]
        for kind in rng.sample(kinds, len(kinds)):
            if kind == "number":
                bad = perturb_number(sent)
            elif kind == "date":
                bad = perturb_date(sent)
            elif kind == "attribution":
                bad = perturb_attribution(sent, [nm for nm in all_names if nm not in names(sent)])
            else:
                bad = perturb_merge(sent, others[0][1]) if others else None
            if bad and bad != sent:
                claims.append({"text": bad, "source_id": src["id"], "label": "false", "error": kind})
                break
    return claims


def claims_report(claims: list[dict]) -> str:
    """A report made of the labelled claims, one bullet each, citing their source."""
    lines = [f"- {c['text'].rstrip('.')}. [{c['source_id']}]" for c in claims]
    return "## Key Findings\n" + "\n".join(lines) + "\n"


def score(claims: list[dict], flagged: list[bool]) -> dict:
    """caught = injected errors flagged; false_alarms = true claims flagged."""
    false_idx = [i for i, c in enumerate(claims) if c["label"] == "false"]
    true_idx = [i for i, c in enumerate(claims) if c["label"] == "true"]
    by_error = {}
    for i in false_idx:
        e = claims[i]["error"]
        hit, total = by_error.get(e, (0, 0))
        by_error[e] = (hit + int(flagged[i]), total + 1)
    return {
        "claims": len(claims),
        "errors": len(false_idx),
        "caught": round(sum(flagged[i] for i in false_idx) / len(false_idx), 3) if false_idx else None,
        "false_alarms": round(sum(flagged[i] for i in true_idx) / len(true_idx), 3) if true_idx else None,
        "caught_by_error": {e: round(h / t, 3) for e, (h, t) in sorted(by_error.items())},
    }


def _grams(text: str, size: int = 4) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def mentioned(feedback: str, claims: list[dict]) -> list[bool]:
    """For the single YES/NO critic: which claims its problem list refers to. Uses the 4-word runs unique to each
    claim, since a merged (false) claim shares text with the true claim it was built from."""
    fb = " ".join(re.findall(r"\w+", feedback.lower()))
    grams = [_grams(c["text"]) for c in claims]
    out = []
    for i, g in enumerate(grams):
        others = set().union(*(grams[:i] + grams[i + 1:])) if len(grams) > 1 else set()
        # A claim whose words all appear inside another claim (a true claim a "merge" error was built from) can't
        # be told apart from it in free text: don't count it, rather than invent a false alarm.
        unique = g - others
        out.append(any(x in fb for x in unique))
    return out


async def evaluate_critic(config: Config, sources: list[dict], mode: str, n: int = 16, seed: int = 7) -> dict:
    """Runs one critic setup over the same labelled claims and scores it."""
    from app import metering
    from app.agents import CriticAgent
    from app.critics import run_panel

    claims = build_labelled_claims(sources, n=n, seed=seed)
    if not claims:
        raise ValueError("no usable sentences in the sources to build claims from")
    report = claims_report(claims)
    counters = metering.start()
    if mode == "single":
        passed, feedback = await CriticAgent(config).run(report, sources)
        flagged = [False] * len(claims) if passed else mentioned(feedback, claims)
    elif mode in ("panel", "debate"):
        result = await run_panel(config, report, sources, debate=(mode == "debate"))
        verdicts = {c["text"][:60]: c["decision"] != "keep" for c in result.claims}
        flagged = [next((v for k, v in verdicts.items() if k[:40] in c["text"] or c["text"][:40] in k), False)
                   for c in claims]
    else:
        raise ValueError(f"unknown critic mode {mode!r}")
    return {**score(claims, flagged), "mode": mode, "llm_calls": counters["llm_calls"],
            "tokens": counters["input_tokens"] + counters["output_tokens"],
            "claims_detail": [{**c, "flagged": f} for c, f in zip(claims, flagged)]}
