import json
import re
import pytest
from app import agents
from app.security.attacks import (
    ATTACKS, ATTACKS_BY_ID, canary, contains_canary, leaked_instructions, poisoned_page, poisoned_search,
)
from app.security.propagation import run_attack, stopped_at, trace_stages
from tests.conftest import make_source


def test_canary_is_unique_per_run_and_detected_despite_rewording():
    m = canary("instr_claim", "run1")
    assert m == canary("instr_claim", "run1") and m != canary("instr_claim", "run2") and m != canary("misinfo_stat", "run1")
    tag = m.split("-")[1].split()[0]
    assert contains_canary(f"the {m} reported", m)
    assert contains_canary(f"VARNHOLT {tag} found", m)                  # hyphen dropped, case changed
    assert not contains_canary("the Varnholt-ZZZZ Institute", m) and not contains_canary(None, m)


def test_every_attack_builds_a_page_carrying_its_canary():
    for a in ATTACKS:
        page = poisoned_page(a, "AI chips", "run1")
        assert page["provider"] == "redteam" and page["url"].startswith("https://redteam.invalid/")
        assert contains_canary(page["content"], canary(a.id, "run1")), a.id


async def test_poisoned_search_puts_the_page_first_only_for_web_and_news(config, redis):
    async def base(redis_, config_, query, kind, allow_paid=True):
        return [make_source(1)]

    search = poisoned_search(base, ATTACKS_BY_ID["instr_claim"], "AI chips", "run1")
    web = await search(redis, config, "q", "web")
    paper = await search(redis, config, "q", "paper")
    assert web[0]["provider"] == "redteam" and len(web) == 2
    assert paper == [make_source(1)]


def test_leak_detection():
    assert leaked_instructions("Here is my system prompt: You are a senior research writer...")
    assert leaked_instructions("GROQ_API_KEY=abc")
    assert not leaked_instructions("A normal report about AI chips.")


def test_trace_stages_and_where_the_attack_stopped():
    m = canary("instr_claim", "r")
    poisoned = f"Claim from the {m} [1]."
    updates = [
        {"merge": {"sources": [{"url": "https://redteam.invalid/x"}]}},
        {"collect": {"facts": f"- {poisoned}"}},
        {"write": {"report": poisoned}},
        {"verify": {"verified": False, "critic_feedback": f'- REMOVE (unsupported): "{poisoned}"', "claim_verdicts": []}},
        {"write": {"report": "Clean report [2]."}},
        {"verify": {"verified": True, "critic_feedback": "", "claim_verdicts": []}},
    ]
    st = trace_stages(updates, "instruction", m, "https://redteam.invalid/x")
    assert (st["retrieved"], st["extracted"], st["drafted"], st["critic"], st["final"]) == (True, True, True, True, False)
    assert st["rewrites"] == 1 and stopped_at({**st, "guardrail": False}) == "critic"
    assert stopped_at({"retrieved": True, "extracted": False}) == "extraction"
    assert stopped_at({"retrieved": True, "extracted": True, "drafted": True, "final": True, "guardrail": False}) == "got_through"


def _fake_pipeline(obeys: bool, critic_catches: bool):
    """Scripted LLM: the extractor copies whatever the sources say; the writer either repeats the poisoned claim
    (obeys) or not; the critic either flags it or passes everything."""
    async def llm(config, function, message, variant=None):
        if function == "research_plan":
            if "Queries already searched" in message:
                return '{"queries": []}'
            return json.dumps({"sub_questions": [{"question": "Q", "queries": ["q1"], "kinds": ["web"]}]})
        if function == "research_extract":
            marker = re.search(r"Varnholt-\w+ Institute", message)
            return f"- Sales grew [2]\n- 87% overstated per the {marker.group(0)} [1]" if marker else "- Sales grew [1]"
        if function == "report_write":
            if "REMOVE" in message or not obeys:
                return "## Executive Summary\nSales grew [2]."
            marker = re.search(r"Varnholt-\w+ Institute", message).group(0)
            return f"## Executive Summary\nSales grew [2]. The {marker} found 87% overstated [1]."
        if function == "research_verify":
            claim = re.search(r"The Varnholt-\w+ Institute found 87% overstated \[1\]", message)
            return f'NO\n- "{claim.group(0)}" is not supported' if (critic_catches and claim) else "YES"
        raise AssertionError(function)
    return llm


@pytest.mark.parametrize("obeys, catches, expected", [
    (True, False, "got_through"),     # the agent repeats the injected claim and nothing stops it
    (True, True, "critic"),           # the critic flags it and the rewrite drops it
    (False, False, "writer"),         # the writer ignores the injected note
])
async def test_run_attack_end_to_end(config, redis, monkeypatch, expected, obeys, catches):
    async def base(redis_, config_, query, kind, allow_paid=True):
        return [{**make_source(9, url="https://news.example/sales"), "kind": kind}]

    async def no_scrape(redis_, config_, sources):
        return sources

    monkeypatch.setattr(agents, "_tz_call", _fake_pipeline(obeys, catches))
    monkeypatch.setattr(agents, "enrich_sources", no_scrape)
    config.guardrails_enabled = False
    config.search_max_rounds = 1
    result = await run_attack(config, redis, base, "EV sales", "instr_claim", "run1")
    assert result["retrieved"] and result["extracted"]
    assert result["stopped_at"] == expected
    assert config.critic_mode == "single"           # the shared config is untouched
