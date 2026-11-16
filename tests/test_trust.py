from app.agents import CRITIC_UNAVAILABLE
from app.history import history_record
from app.trust import label_for, trust_score


def verdicts(kept, flagged, agreed=True):
    return ([{"decision": "keep", "agreed": agreed}] * kept) + ([{"decision": "fix", "agreed": False}] * flagged)


def test_score_is_the_share_of_claims_both_critics_kept():
    t = trust_score({"live_sources": True, "claim_verdicts": verdicts(18, 2), "iterations": 2}, sources_count=9)
    assert t["score"] == 90 and t["label"] == "High"
    assert (t["claims_checked"], t["claims_verified"], t["claims_flagged"]) == (20, 18, 2)
    assert t["critics_agreed"] == 18 and t["sources"] == 9 and t["drafts"] == 2


def test_labels():
    assert [label_for(s) for s in (100, 90, 89, 70, 69, 0)] == ["High", "High", "Medium", "Medium", "Low", "Low"]


def test_reports_without_a_per_claim_check_get_no_score_and_say_why():
    assert "model knowledge" in trust_score({"live_sources": False})["reason"]
    assert "could not complete" in trust_score({"live_sources": True, "critic_feedback": CRITIC_UNAVAILABLE})["reason"]
    assert "single-critic" in trust_score({"live_sources": True, "claim_verdicts": [], "verified": True})["reason"]
    rewritten = trust_score({"live_sources": True, "claim_verdicts": verdicts(5, 0)}, safety_rewritten=True)
    assert rewritten["score"] is None and "safety" in rewritten["reason"]


def test_history_keeps_the_score():
    trust = {"score": 90, "label": "High"}
    row = history_record("j1", "s1", {"status": "done", "report": "## Summary\nText.", "trust": trust})
    assert row["trust"] == trust
