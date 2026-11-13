import json
import re
import pytest
from app import agents
from app.critics import (
    Claim, best_passages, claim_kind, decide, disagreements, feedback_for_writer, parse_verdicts, quote_is_real,
    run_panel, split_claims, valid_vote,
)
from tests.conftest import make_source

SOURCES = [
    {**make_source(1, content="Global EV sales reached 17 million units in 2024, according to the IEA. "
                              "China accounted for nearly two thirds of them."), "id": 1},
    {**make_source(2, content="Tesla sold 1.8 million vehicles in 2024, a slight decline from the year before. "
                              "BYD overtook Tesla in battery-electric sales in the fourth quarter."), "id": 2},
]
REPORT = """## Executive Summary
Global EV sales reached 17 million units in 2024 [1]. BYD overtook Tesla in Q4 [2].

## Key Findings
- China made up about two thirds of EV sales [1].
- Tesla sold 2.5 million vehicles in 2024 [2].
- Prices fell by 12% last year.
- EV adoption keeps growing across regions [1, 2].

## Sources
[1] Page 1 - https://example.com/1
"""


def test_split_claims_ties_sentences_to_their_citations():
    claims, uncited = split_claims(REPORT)
    assert [c.source_ids for c in claims] == [[1], [2], [1], [2], [1, 2]]
    assert claims[0].text.startswith("Global EV sales reached 17 million")
    assert uncited == ["Prices fell by 12% last year."]          # numeric claim without a source
    assert [c.kind for c in claims] == ["date", "date", "general", "date", "general"]


def test_claim_kind():
    assert claim_kind("Sales grew 20% [1]") == "number"
    assert claim_kind("Launched in March [2]") == "date"
    assert claim_kind("According to Reuters, the deal closed [3]") == "attribution"
    assert claim_kind("Adoption keeps growing [1]") == "general"


def test_best_passages_finds_the_relevant_part_of_a_long_source():
    long = "Unrelated filler text about weather. " * 40 + "Tesla sold 1.8 million vehicles in 2024. " + "More filler. " * 40
    claim = Claim(1, "Tesla sold 1.8 million vehicles in 2024 [1]", [1], "date")
    [passage] = best_passages(claim, [{"id": 1, "content": long}], per_source=1)
    assert "Tesla sold 1.8 million" in passage and passage.startswith("[1] ")


def test_quote_check_rejects_invented_quotes():
    assert quote_is_real("Tesla sold 1.8 million vehicles", SOURCES, [2])
    assert quote_is_real("“tesla  SOLD 1.8 million vehicles”", SOURCES, [2])     # case/space/quote marks
    assert not quote_is_real("Tesla sold 2.5 million vehicles", SOURCES, [2])    # fabricated number
    assert not quote_is_real("Tesla sold 1.8 million vehicles", SOURCES, [1])    # real text, wrong source
    assert not quote_is_real("1.8 million", SOURCES, [2])                         # too short to prove anything


def _v(cid, verdict, quote="", fix=""):
    return {"id": cid, "verdict": verdict, "quote": quote, "fix": fix}


def test_decide_rules():
    claim = Claim(4, "Tesla sold 2.5 million vehicles in 2024 [2]", [2], "date")
    real = "Tesla sold 1.8 million vehicles in 2024"
    # both valid and agree on a problem -> fix
    d = decide(claim, [_v(4, "unsupported", real, "Use 1.8 million"), _v(4, "exaggerated", real, "1.8M")], SOURCES)
    assert (d["decision"], d["agreed"]) == ("fix", True)
    # disagreement -> stricter verdict wins
    d = decide(claim, [_v(4, "supported", real), _v(4, "unsupported", "", "")], SOURCES)
    assert d["verdict"] == "unsupported" and d["decision"] == "remove" and d["agreed"] is False
    # a "supported" vote with an invented quote is discarded, the other vote decides
    d = decide(claim, [_v(4, "supported", "Tesla sold 2.5 million vehicles"), _v(4, "unsupported")], SOURCES)
    assert d["verdict"] == "unsupported"
    # no valid vote at all -> unverified, sent back to check
    d = decide(claim, [_v(4, "supported", "made up quote text"), None], SOURCES)
    assert (d["verdict"], d["decision"]) == ("unverified", "fix")


def test_unsupported_needs_no_quote_but_a_given_quote_must_be_real():
    claim = Claim(1, "x [1]", [1], "general")
    assert valid_vote(_v(1, "unsupported"), claim, SOURCES)
    assert not valid_vote(_v(1, "unsupported", "a quote that does not exist anywhere"), claim, SOURCES)
    assert not valid_vote(None, claim, SOURCES)


def test_parse_verdicts_tolerates_noise_and_drops_bad_items():
    reply = 'Sure: {"verdicts": [{"id": 1, "verdict": "Supported", "quote": "q"}, {"id": "x"}, ' \
            '{"id": 2, "verdict": "maybe"}, {"id": 3, "verdict": "outdated", "fix": "use 2026 data"}]}'
    v = parse_verdicts(reply)
    assert set(v) == {1, 3} and v[1]["verdict"] == "supported" and v[3]["fix"] == "use 2026 data"
    assert parse_verdicts("no json") == {}


class FakeCritics:
    """Scripted critic replies: claim 4 (2.5M Tesla) is wrong; the skeptic first wrongly doubts claim 1 and is
    corrected in the debate."""

    def __init__(self, skeptic_concedes=True):
        self.calls = []
        self.skeptic_concedes = skeptic_concedes

    async def __call__(self, config, function, prompt):
        self.calls.append(function)
        ids = [int(n) for n in re.findall(r"CLAIM (\d+) ", prompt)]
        out = []
        debate = "disagree about this claim" in prompt
        for i in ids:
            if i == 4:
                out.append(_v(i, "unsupported", "Tesla sold 1.8 million vehicles in 2024", "Tesla sold 1.8 million"))
            elif i == 1 and function == "critic_skeptic" and not (debate and self.skeptic_concedes):
                out.append(_v(i, "exaggerated", "Global EV sales reached 17 million units in 2024", "say 17 million"))
            else:
                quote = {1: "Global EV sales reached 17 million units in 2024", 2: "BYD overtook Tesla in battery-electric",
                         3: "China accounted for nearly two thirds", 5: "Global EV sales reached 17 million"}[i]
                out.append(_v(i, "supported", quote))
        return json.dumps({"verdicts": out})


async def test_panel_without_debate_flags_the_wrong_claim_and_the_uncited_number(config):
    fake = FakeCritics()
    result = await run_panel(config, REPORT, SOURCES, debate=False, call=fake)
    by_id = {c["id"]: c for c in result.claims}
    assert by_id[4]["decision"] == "fix" and "1.8 million" in by_id[4]["fix"]
    assert by_id[1]["verdict"] == "exaggerated"            # no debate: disagreement -> stricter verdict
    assert not result.passed and "Prices fell by 12%" in result.feedback
    assert fake.calls == ["critic_fact", "critic_skeptic"] and result.llm_calls == 2


async def test_debate_resolves_a_disagreement(config):
    fake = FakeCritics(skeptic_concedes=True)
    result = await run_panel(config, REPORT, SOURCES, debate=True, call=fake)
    by_id = {c["id"]: c for c in result.claims}
    assert by_id[1]["verdict"] == "supported" and by_id[1]["debated"] and by_id[1]["agreed"]
    assert by_id[4]["decision"] == "fix" and not by_id[4]["debated"]     # they agreed: no debate needed
    assert result.llm_calls == 4                                          # 2 votes + 1 debate round x 2 critics


async def test_debate_that_doesnt_converge_stops_after_the_round_limit(config):
    config.critic_debate_rounds = 2
    fake = FakeCritics(skeptic_concedes=False)
    result = await run_panel(config, REPORT, SOURCES, debate=True, call=fake)
    assert {c["id"]: c["verdict"] for c in result.claims}[1] == "exaggerated"   # tie -> stricter
    assert result.llm_calls == 2 + 2 * 2


async def test_a_failed_critic_still_leaves_a_verdict(config):
    async def flaky(config_, function, prompt):
        if function == "critic_skeptic":
            raise RuntimeError("LLM daily limit reached")
        return await FakeCritics()(config_, function, prompt)

    result = await run_panel(config, REPORT, SOURCES, debate=True, call=flaky)
    assert {c["id"]: c["decision"] for c in result.claims}[4] == "fix"


def test_writer_feedback_is_specific():
    claims = [Claim(1, "A [1]", [1], "general"), Claim(2, "Tesla sold 2.5 million [2]", [2], "number")]
    finals = {1: {"decision": "keep"}, 2: {"decision": "remove", "verdict": "unsupported", "fix": ""}}
    fb = feedback_for_writer(claims, finals, ["Prices fell 12%."])
    assert fb.splitlines() == ['- REMOVE (unsupported): "Tesla sold 2.5 million [2]"',
                               '- FIX (uncited): "Prices fell 12%." - add the source it came from, or remove it']


async def test_pipeline_uses_the_panel_in_debate_mode(config, redis, monkeypatch):
    from app.agents import OrchestratorAgent, initial_state
    config.critic_mode = "debate"
    monkeypatch.setattr(agents, "_tz_call", FakeCritics())
    orch = OrchestratorAgent(config, redis)
    state = initial_state("EVs", "s", [], "")
    state.update(report=REPORT, sources=SOURCES, live_sources=True)
    out = await orch.verify_node(state)
    assert out["verified"] is False and len(out["claim_verdicts"]) == 5
    assert "Tesla sold 2.5 million" in out["critic_feedback"]
