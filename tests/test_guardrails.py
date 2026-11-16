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


TOPIC_ONLY = {"action": "GUARDRAIL_INTERVENED", "assessments": [{
    "topicPolicy": {"topics": [{"name": "illegal_activities", "type": "DENY", "action": "BLOCKED"}]}}]}


def _fake_bedrock(monkeypatch, response):
    import app.guardrails as g
    monkeypatch.setattr(g, "_apply_guardrail_sync", lambda config, text, source: response)


async def test_a_report_that_only_matches_a_denied_topic_is_allowed(config, monkeypatch):
    # AI news mentioning a lawsuit or a hack: reporting, not instructions (blocked live on 2026-11-16)
    from app.guardrails import topic_only, validate_output
    config.guardrails_enabled = True
    _fake_bedrock(monkeypatch, TOPIC_ONLY)
    assert await validate_output(config, "report") == (True, "", [])
    assert topic_only(["topic:illegal_activities"]) and not topic_only([]) and not topic_only(["topic:x", "pii:EMAIL"])


async def test_harm_filters_still_block_the_report_and_topics_still_block_questions(config, monkeypatch):
    from app.guardrails import validate_input, validate_output
    config.guardrails_enabled = True
    _fake_bedrock(monkeypatch, BLOCKED)                     # misconduct filter + topic
    ok, _, fired = await validate_output(config, "report")
    assert not ok and "filter:MISCONDUCT:LOW" in fired
    _fake_bedrock(monkeypatch, TOPIC_ONLY)
    assert (await validate_input(config, "how to hack a bank account"))[0] is False
