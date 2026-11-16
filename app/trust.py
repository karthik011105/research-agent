"""Trust score for a finished report (code only, no LLM).

The two-critic panel gives every claim of the final draft a verdict against its sources (app/critics.py). The trust
score is about the report's FACTS: the share of factual claims that the critics' final decision kept.

  facts            claims outside the interpretation sections: the Executive Summary, Key Findings and any
                   other section of findings (numbers, dates, names, events, each with its [n])
  interpretations  claims in Analysis, Discussion, Conclusion, Outlook, Implications or Recommendations: the
                   writer's reasoning about the facts. They are checked too and their flags are shown, but they are
                   counted separately - "this suggests a growing market" is not wrong the way a wrong number is
  score            facts kept / facts checked, as a percentage (all claims, if a report has no factual ones)
  label            High (>= 90) · Medium (>= 70) · Low (below 70)

A report gets no score - with the reason shown instead - when there was nothing to check it against: no live
sources, the fact-check could not finish, single-critic mode (one YES/NO verdict, no per-claim check), or a safety
rewrite after the check changed the text the critics saw.
"""
import re

HIGH, MEDIUM = 90, 70
_HEADING = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*(.+?)|\*\*(.+?)\*\*:?)[ \t]*$", re.M)
_SECTION = re.compile(r"summary|key findings|findings|analysis|discussion|conclusion|outlook|implication|"
                      r"recommendation|what to watch", re.I)
_INTERPRETATION = re.compile(r"analysis|discussion|conclusion|outlook|implication|recommendation|what to watch", re.I)


def label_for(score: int) -> str:
    return "High" if score >= HIGH else "Medium" if score >= MEDIUM else "Low"


def unscored(reason: str) -> dict:
    return {"score": None, "label": "Not scored", "reason": reason}


def is_interpretation(report: str, claim_text: str) -> bool:
    """Is the claim in one of the report's interpretation sections? Decided by the last top-level section heading
    (Executive Summary, Key Findings, Analysis, ...) before it; sub-question headings inside a section don't count."""
    pos = report.find((claim_text or "")[:80])
    if pos < 0:
        return False
    section = ""
    for m in _HEADING.finditer(report[:pos]):
        title = (m.group(1) or m.group(2) or "").strip()
        if _SECTION.search(title) and len(title) < 40:
            section = title
    return bool(_INTERPRETATION.search(section))


def trust_score(state: dict, sources_count: int = 0, safety_rewritten: bool = False, report: str | None = None) -> dict:
    from app.agents import CRITIC_UNAVAILABLE
    if not state.get("live_sources"):
        return unscored("written from model knowledge, with no live sources to check against")
    if state.get("critic_feedback") == CRITIC_UNAVAILABLE:
        return unscored("the fact-check could not complete for this report")
    verdicts = state.get("claim_verdicts") or []
    if not verdicts:
        return unscored("single-critic mode gives one verdict for the whole report, not per claim")
    if safety_rewritten:
        return unscored("the report was rewritten for safety after the fact-check")
    text = report if report is not None else state.get("report", "")
    in_reasoning = [is_interpretation(text, v.get("text", "")) for v in verdicts]
    reasoning = [v for v, r in zip(verdicts, in_reasoning) if r]
    facts = [v for v, r in zip(verdicts, in_reasoning) if not r] or verdicts
    kept = sum(v.get("decision") == "keep" for v in facts)
    score = round(100 * kept / len(facts))
    return {
        "score": score,
        "label": label_for(score),
        "claims_checked": len(facts),
        "claims_verified": kept,
        "claims_flagged": len(facts) - kept,
        "interpretations_checked": len(reasoning),
        "interpretations_flagged": sum(v.get("decision") != "keep" for v in reasoning),
        "critics_agreed": sum(bool(v.get("agreed")) for v in facts),
        "sources": sources_count,
        "drafts": int(state.get("iterations") or 1),
    }


def needs_review(trust: dict, notes: list) -> bool:
    """Human-in-the-loop: a new report a person should look at before it is reused for anyone else - its trust score
    is below High, it has fact-check notes, or it could not be scored at all."""
    return bool(notes) or trust.get("score") is None or trust["score"] < HIGH
