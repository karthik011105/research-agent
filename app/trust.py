"""Trust score for a finished report (code only, no LLM).

The two-critic panel gives every claim of the final draft a verdict against its sources (app/critics.py). The trust
score is the share of checked claims both critics' final decision kept:

  score  = claims kept / claims checked, as a percentage
  label  = High (>= 90) · Medium (>= 70) · Low (below 70)

A report gets no score - with the reason shown instead - when there was nothing to check it against: no live
sources, the fact-check could not finish, single-critic mode (one YES/NO verdict, no per-claim check), or a safety
rewrite after the check changed the text the critics saw.
"""
HIGH, MEDIUM = 90, 70


def label_for(score: int) -> str:
    return "High" if score >= HIGH else "Medium" if score >= MEDIUM else "Low"


def unscored(reason: str) -> dict:
    return {"score": None, "label": "Not scored", "reason": reason}


def trust_score(state: dict, sources_count: int = 0, safety_rewritten: bool = False) -> dict:
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
    checked = len(verdicts)
    kept = sum(v.get("decision") == "keep" for v in verdicts)
    score = round(100 * kept / checked)
    return {
        "score": score,
        "label": label_for(score),
        "claims_checked": checked,
        "claims_verified": kept,
        "claims_flagged": checked - kept,
        "critics_agreed": sum(bool(v.get("agreed")) for v in verdicts),
        "sources": sources_count,
        "drafts": int(state.get("iterations") or 1),
    }


def needs_review(trust: dict, notes: list) -> bool:
    """Human-in-the-loop: a new report a person should look at before it is reused for anyone else - its trust score
    is below High, it has fact-check notes, or it could not be scored at all."""
    return bool(notes) or trust.get("score") is None or trust["score"] < HIGH
