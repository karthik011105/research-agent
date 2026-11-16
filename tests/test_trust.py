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


# ── human-in-the-loop review ─────────────────────────────────────────────────
from app.history import review_report
from app.trust import needs_review


def test_doubtful_reports_need_a_human_review():
    assert not needs_review({"score": 95}, [])
    assert needs_review({"score": 85}, [])                       # below High
    assert needs_review({"score": 95}, ["a flagged claim"])      # fact-check notes
    assert needs_review({"score": None}, [])                     # nothing could be checked


def test_history_records_the_review_status():
    done = {"status": "done", "report": "## Summary\nText."}
    assert history_record("j", "s", {**done, "review": {"status": "pending"}})["review_status"] == "pending"
    assert history_record("j", "s", done)["review_status"] == "not_needed"


class FakeStore:
    def __init__(self):
        self.rows = {"held": {"job_id": "held", "review_status": "pending"}, "fine": {"job_id": "fine", "review_status": "not_needed"}}

    async def decide(self, job_id, status, note):
        row = self.rows.get(job_id)
        if not row or row["review_status"] != "pending":
            return None
        row.update(review_status=status, review_note=note)
        return dict(row)


async def test_only_an_approved_report_enters_shared_memory():
    remembered = []

    async def remember(row):
        remembered.append(row["job_id"])
    store = FakeStore()
    rejected = await review_report(store, "held", "reject", "  wrong figure  ", remember)
    assert rejected["review_status"] == "rejected" and rejected["review_note"] == "wrong figure" and remembered == []
    store = FakeStore()
    approved = await review_report(store, "held", "approve", "", remember)
    assert approved["review_status"] == "approved" and remembered == ["held"]
    assert await review_report(store, "held", "reject", "", remember) is None       # already decided
    assert await review_report(store, "fine", "approve", "", remember) is None      # never needed a review
