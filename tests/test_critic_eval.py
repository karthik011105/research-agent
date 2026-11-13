import json
import re
from app import agents, critic_eval
from app.critic_eval import (
    build_labelled_claims, claims_report, evaluate_critic, perturb_attribution, perturb_date, perturb_merge,
    perturb_number, score,
)
from app.critics import split_claims

SOURCES = [
    {"id": 1, "title": "EV report", "url": "https://a.example/1", "content":
        "Global electric vehicle sales reached 17 million units in 2024 according to the International Energy Agency. "
        "China accounted for 11 million of those sales, driven by aggressive pricing from domestic makers. "
        "European sales were flat at 3.2 million units as subsidies were withdrawn in Germany."},
    {"id": 2, "title": "Tesla results", "url": "https://b.example/2", "content":
        "Tesla delivered 1.8 million vehicles in 2024, a small decline compared with the previous year. "
        "Analysts at Morgan Stanley expect deliveries to grow by 15 percent in 2025 as new models launch. "
        "BYD Company overtook Tesla in battery electric sales during the final quarter of 2024."},
]


def test_perturbations_change_exactly_the_targeted_fact():
    s = "China accounted for 11 million of those sales in 2024."
    assert perturb_number(s) == "China accounted for 16 million of those sales in 2024."   # x1.5, year untouched
    assert perturb_date(s) == "China accounted for 11 million of those sales in 2026."
    assert perturb_attribution("Analysts at Morgan Stanley expect growth.", ["Goldman Sachs"]) == \
        "Analysts at Goldman Sachs expect growth."
    assert perturb_merge("Sales rose.", "Prices fell sharply, analysts said.") == "Sales rose, and prices fell sharply."
    assert perturb_number("No figures here.") is None and perturb_date("No year here.") is None


def test_labelled_set_is_balanced_deterministic_and_cites_its_source():
    a = build_labelled_claims(SOURCES, n=6, seed=1)
    b = build_labelled_claims(SOURCES, n=6, seed=1)
    assert a == b and len(a) == 6
    assert {c["label"] for c in a} == {"true", "false"}
    for c in a:
        source_text = next(s["content"] for s in SOURCES if s["id"] == c["source_id"])
        assert (c["text"] in source_text) == (c["label"] == "true")      # true = verbatim, false = altered


def test_claims_report_splits_back_into_the_same_claims():
    claims = build_labelled_claims(SOURCES, n=6, seed=1)
    parsed, uncited = split_claims(claims_report(claims))
    assert len(parsed) == len(claims) and uncited == []
    assert [p.source_ids for p in parsed] == [[c["source_id"]] for c in claims]


def test_score_counts_catches_and_false_alarms():
    claims = [{"label": "true", "error": None}, {"label": "false", "error": "number"},
              {"label": "false", "error": "date"}, {"label": "true", "error": None}]
    s = score(claims, [True, True, False, False])
    assert (s["caught"], s["false_alarms"], s["errors"]) == (0.5, 0.5, 2)
    assert s["caught_by_error"] == {"date": 0.0, "number": 1.0}


async def test_evaluate_panel_flags_injected_errors(config, monkeypatch):
    claims = build_labelled_claims(SOURCES, n=6, seed=1)
    truth = {c["text"].rstrip(".")[:40]: c["label"] for c in claims}

    async def oracle(config_, function, prompt):
        # a perfect critic: supports verbatim claims (quoting them), rejects altered ones
        out = []
        for cid, text in re.findall(r"CLAIM (\d+) \(type: \w+\): (.+)", prompt):
            label = next(v for k, v in truth.items() if text.startswith(k[:30]))
            if label == "true":
                out.append({"id": int(cid), "verdict": "supported", "quote": re.sub(r"\s*\[\d+\]\.?$", "", text).rstrip(".")[:80]})
            else:
                out.append({"id": int(cid), "verdict": "unsupported", "quote": "", "fix": "match the source"})
        return json.dumps({"verdicts": out})

    monkeypatch.setattr(agents, "_tz_call", oracle)
    result = await evaluate_critic(config, SOURCES, "debate", n=6, seed=1)
    assert result["caught"] == 1.0 and result["false_alarms"] == 0.0
    assert result["llm_calls"] == 0          # the fake doesn't go through metering; real calls are counted


async def test_evaluate_single_critic_uses_its_problem_list(config, monkeypatch):
    claims = build_labelled_claims(SOURCES, n=6, seed=1)
    wrong = [c for c in claims if c["label"] == "false"]

    async def single(config_, function, prompt):
        assert function == "research_verify"
        return "NO\n" + "\n".join(f'- "{c["text"]}" does not match the source' for c in wrong[:1])

    monkeypatch.setattr(agents, "_tz_call", single)
    result = await evaluate_critic(config, SOURCES, "single", n=6, seed=1)
    assert result["caught"] == round(1 / len(wrong), 3) and result["false_alarms"] == 0.0
