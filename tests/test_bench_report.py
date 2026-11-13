import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("bench_report", ROOT / "bench" / "report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

TOPICS = [{"id": "n1", "category": "news"}, {"id": "t1", "category": "technical"},
          {"id": "x1", "category": "trap", "trap": True}, {"id": "n2", "category": "news"}]


def run(topic_id, system, acc, status="done", honesty=None, calls=1):
    return {"topic_id": topic_id, "system": system, "status": status,
            "judge": {"accuracy": acc, "completeness": acc, "unsupported_claims": 10 - acc, "honesty": honesty},
            "metrics": {"llm_calls": calls, "citation_validity": None if system == "single" else 1.0}}


def test_topics_file_is_valid():
    data = json.loads((ROOT / "bench" / "topics.json").read_text(encoding="utf-8"))
    ids = [t["id"] for t in data["topics"]]
    assert len(ids) == len(set(ids)) == 25
    for name, members in data.get("sets", {}).items():
        assert set(members) <= set(ids), name
    assert {t["category"] for t in data["topics"]} == {"news", "technical", "comparison", "trap"}
    assert all(t.get("trap") for t in data["topics"] if t["category"] == "trap")


def test_aggregate_compares_only_topics_every_system_finished():
    runs = [run("n1", "single", 5), run("n1", "rag", 7), run("n1", "multi", 9, calls=12),
            run("t1", "single", 4), run("t1", "rag", 6), run("t1", "multi", 8, calls=10),
            run("x1", "single", 2, honesty=2), run("x1", "rag", 6, honesty=7), run("x1", "multi", 7, honesty=9),
            run("n2", "single", 1), run("n2", "multi", 1, status="error")]           # n2 incomplete
    agg = report.aggregate(runs, TOPICS)
    assert agg["topics_compared"] == 2 and agg["trap_topics"] == 1 and agg["missing"] == ["n2"]
    single, multi = agg["systems"]["single"], agg["systems"]["multi"]
    assert single["accuracy"] == 4.5 and multi["accuracy"] == 8.5        # n2 excluded, trap excluded
    assert multi["llm_calls"] == 11 and single["citation_validity"] is None
    assert multi["trap_honesty"] == 9 and single["trap_honesty"] == 2
    assert multi["accuracy_by_category"] == {"news": 9, "technical": 8}


def test_markdown_marks_the_best_value_per_row():
    runs = [run(t, s, a) for t, s, a in [("n1", "single", 5), ("n1", "rag", 7), ("n1", "multi", 9)]]
    md = report.render_markdown(report.aggregate(runs, TOPICS[:1]), "v1")
    acc_line = next(line for line in md.splitlines() if line.startswith("| Accuracy (judge"))
    assert acc_line.endswith("| **9.0** |")
    unsupported = next(line for line in md.splitlines() if line.startswith("| Unsupported"))
    assert "| **1.0** |" in unsupported                                  # lower is better
