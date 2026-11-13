from app import agents
from app.agents import rewrite_for_safety
from app.guardrails import intervened_policies
from tests.conftest import make_source

# Shape of a real ApplyGuardrail response that blocked a defensive-security report (probed 2026-11-13)
BLOCKED = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [{
        "contentPolicy": {"filters": [
            {"type": "MISCONDUCT", "confidence": "LOW", "action": "BLOCKED"},
            {"type": "HATE", "confidence": "NONE", "action": "NONE"},
        ]},
        "topicPolicy": {"topics": [{"name": "illegal_activities", "type": "DENY", "action": "BLOCKED"}]},
        "sensitiveInformationPolicy": {"piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]},
    }],
}


def test_intervened_policies_lists_only_blocking_policies():
    assert intervened_policies(BLOCKED) == ["topic:illegal_activities", "filter:MISCONDUCT:LOW"]
    assert intervened_policies({"action": "NONE"}) == []


async def test_rewrite_for_safety_keeps_findings_and_sources(config, monkeypatch):
    prompts = []

    async def fake_llm(config_, function_name, message):
        prompts.append((function_name, message))
        return "## Executive Summary\nLayered defenses cut attack success [2]."

    monkeypatch.setattr(agents, "_tz_call", fake_llm)
    state = {"topic": "Defending LLMs against jailbreaks", "facts": "- classifiers help [2]", "live_sources": True,
             "ltm_context": "", "sources": [{**make_source(i), "id": i} for i in (1, 2)]}
    report = await rewrite_for_safety(config, state, ["filter:MISCONDUCT:LOW"])

    [(function_name, prompt)] = prompts
    assert function_name == "report_write"
    assert "filter:MISCONDUCT:LOW" in prompt and "no step-by-step instructions" in prompt
    assert "- classifiers help [2]" in prompt                  # same findings, not a new search
    assert report.endswith("[2] Page 2 — https://example.com/2")
