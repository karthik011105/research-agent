from app.agents import CRITIC_UNAVAILABLE
from app.history import history_record
from app.trust import label_for, trust_score


def verdicts(kept, flagged, agreed=True, verdict="misattributed"):
    return ([{"decision": "keep", "verdict": "supported", "agreed": agreed}] * kept) +            ([{"decision": "fix", "verdict": verdict, "agreed": False}] * flagged)


def test_score_is_the_share_of_facts_not_contradicted_by_their_sources():
    t = trust_score({"live_sources": True, "claim_verdicts": verdicts(18, 2), "iterations": 2}, sources_count=9)
    assert t["score"] == 90 and t["label"] == "High" and t["basis"] == "not_contradicted"
    assert (t["claims_checked"], t["claims_verified"], t["claims_contradicted"], t["claims_unconfirmed"]) == (20, 18, 2, 0)
    assert t["critics_agreed"] == 18 and t["sources"] == 9 and t["drafts"] == 2


def test_not_found_in_the_excerpts_is_reported_but_is_not_a_contradiction():
    mixed = verdicts(11, 3, verdict="exaggerated") + verdicts(0, 5, verdict="unsupported") + verdicts(0, 1, verdict="unverified")
    t = trust_score({"live_sources": True, "claim_verdicts": mixed})
    assert (t["claims_verified"], t["claims_unconfirmed"], t["claims_contradicted"]) == (11, 6, 3)
    assert t["score"] == 85 and t["label"] == "Medium"


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


REPORT = """## Executive Summary
BYD sold 68% of EVs in Vietnam in H1 2026 [1].

## Key Findings
### What are the latest implications of EV policy changes in Europe for 2026 sales?
- Europe sold 4.5 million EVs by October 2026 [2].

## Analysis
Europe's growth suggests faster adoption outside China [2].

## Conclusion
The market keeps growing [1, 2].
"""


def test_the_score_counts_facts_and_reports_interpretations_separately():
    from app.trust import is_interpretation
    verdicts = [
        {"text": "BYD sold 68% of EVs in Vietnam in H1 2026 [1].", "decision": "fix", "verdict": "misattributed"},
        {"text": "Europe sold 4.5 million EVs by October 2026 [2].", "decision": "keep", "verdict": "supported"},
        {"text": "Europe's growth suggests faster adoption outside China [2].", "decision": "fix", "verdict": "unsupported"},
        {"text": "The market keeps growing [1, 2].", "decision": "fix", "verdict": "unsupported"},
    ]
    assert [is_interpretation(REPORT, v["text"]) for v in verdicts] == [False, False, True, True]
    t = trust_score({"live_sources": True, "claim_verdicts": verdicts}, report=REPORT)
    assert (t["score"], t["claims_checked"], t["claims_verified"]) == (50, 2, 1)
    assert (t["interpretations_checked"], t["interpretations_flagged"]) == (2, 2)


def test_a_report_with_only_interpretations_is_scored_on_all_its_claims():
    only = "## Analysis\nGrowth suggests adoption [1].\n"
    t = trust_score({"live_sources": True, "claim_verdicts": [{"text": "Growth suggests adoption [1].", "decision": "keep"}]},
                    report=only)
    assert t["score"] == 100 and t["claims_checked"] == 1


def test_dates_inside_citations_are_removed():
    from app.agents import normalize_citations
    assert normalize_citations("safeguards [1, 2026‑06‑01; 7, 2025‑10‑25].") == "safeguards [1, 7]."
    assert normalize_citations("reports [2, 2026-05-13; 8, 2026-01-01]") == "reports [2, 8]"
    assert normalize_citations("in [2024] the share") == "in [2024] the share"        # a year alone stays
    assert normalize_citations("grew [3] and [1, 4]") == "grew [3] and [1, 4]"
