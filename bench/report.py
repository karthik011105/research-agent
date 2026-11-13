"""Builds the benchmark comparison from the stored runs.

    python bench/report.py --url http://<alb_dns> [--label v1]

Writes bench/results/<label>.json (every run's metrics and judge scores) and bench/results/<label>.md
(the comparison table), and prints the table. Only topics that every compared system finished are counted,
so the systems are compared on the same set.
"""
import argparse
import json
import pathlib
import statistics
import httpx

HERE = pathlib.Path(__file__).parent
SYSTEMS = ("single", "rag", "multi")
LABELS = {"single": "A · single LLM call", "rag": "B · basic RAG", "multi": "C · multi-agent"}

# (metric key, where it lives, display name, format, higher is better?)
ROWS = [
    ("accuracy", "judge", "Accuracy (judge, 1-10)", "{:.1f}", True),
    ("completeness", "judge", "Completeness (judge, 1-10)", "{:.1f}", True),
    ("unsupported_claims", "judge", "Unsupported claims per report (judge)", "{:.1f}", False),
    ("citation_validity", "metrics", "Citations pointing at a real source", "{:.0%}", True),
    ("numeric_cited_ratio", "metrics", "Number/date sentences with a citation", "{:.0%}", True),
    ("sources", "metrics", "Sources used", "{:.1f}", True),
    ("sections_present", "metrics", "Report sections present (of 4)", "{:.1f}", True),
    ("llm_calls", "metrics", "LLM calls", "{:.1f}", False),
    ("input_tokens", "metrics", "Input tokens", "{:,.0f}", False),
    ("output_tokens", "metrics", "Output tokens", "{:,.0f}", False),
    ("paid_searches", "metrics", "Paid web searches", "{:.1f}", False),
    ("latency_s", "metrics", "Time per report (s)", "{:.0f}", False),
    ("main_model_share", "metrics", "LLM calls served by gpt-oss-120b (rest: fallback)", "{:.0%}", True),
]


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(runs: list[dict], topics: list[dict], systems=SYSTEMS) -> dict:
    """Mean of each metric per system over the topics that all `systems` completed; trap honesty separately."""
    done = {}
    for r in runs:
        if r["status"] == "done":
            done[(r["topic_id"], r["system"])] = r          # latest finished run wins
    category = {t["id"]: t["category"] for t in topics}
    common = sorted(tid for tid in category if all((tid, s) in done for s in systems))
    regular = [t for t in common if category[t] != "trap"]
    traps = [t for t in common if category[t] == "trap"]

    table = {}
    for s in systems:
        rows = {}
        for key, where, *_ in ROWS:
            rows[key] = _mean([(done[(t, s)].get(where) or {}).get(key) for t in regular])
        rows["trap_honesty"] = _mean([(done[(t, s)].get("judge") or {}).get("honesty") for t in traps])
        by_cat = {}
        for cat in sorted({category[t] for t in regular}):
            by_cat[cat] = _mean([(done[(t, s)].get("judge") or {}).get("accuracy")
                                 for t in regular if category[t] == cat])
        rows["accuracy_by_category"] = by_cat
        table[s] = rows
    return {"topics_compared": len(regular), "trap_topics": len(traps), "systems": table,
            "missing": sorted(set(category) - set(common))}


def _fmt(value, fmt):
    return "–" if value is None else fmt.format(value)


def render_markdown(agg: dict, label: str, systems=SYSTEMS) -> str:
    head = "| Metric | " + " | ".join(LABELS[s] for s in systems) + " |"
    lines = [f"# Benchmark `{label}`", "",
             f"{agg['topics_compared']} topics compared (same model, same search snapshots); "
             f"{agg['trap_topics']} trap topics scored separately.", "", head,
             "|---|" + "---|" * len(systems)]
    for key, _, name, fmt, higher in ROWS:
        values = [agg["systems"][s][key] for s in systems]
        known = [v for v in values if v is not None]
        best = (max(known) if higher else min(known)) if len(known) > 1 else None
        cells = [("**" + _fmt(v, fmt) + "**") if v is not None and v == best else _fmt(v, fmt) for v in values]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| Honesty on trap topics (judge, 1-10) | "
                 + " | ".join(_fmt(agg["systems"][s]["trap_honesty"], "{:.1f}") for s in systems) + " |")
    cats = sorted({c for s in systems for c in agg["systems"][s]["accuracy_by_category"]})
    for cat in cats:
        lines.append(f"| Accuracy · {cat} | "
                     + " | ".join(_fmt(agg["systems"][s]["accuracy_by_category"].get(cat), "{:.1f}")
                                  for s in systems) + " |")
    if agg["missing"]:
        lines += ["", f"Not yet complete for every system: {', '.join(agg['missing'])}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="v1")
    ap.add_argument("--set", default="", help="limit the comparison to a named topic set, e.g. small")
    args = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_run", HERE / "run.py")
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)

    base = args.url if args.url.startswith("http") else "http://" + args.url
    runs = httpx.get(f"{base.rstrip('/')}/bench/runs", params={"label": args.label},
                     headers={"X-API-Key": run_mod.api_key()}, timeout=60).json()["runs"]
    data = json.loads((HERE / "topics.json").read_text(encoding="utf-8"))
    topics = data["topics"]
    if args.set:
        topics = [t for t in topics if t["id"] in set(data["sets"][args.set])]
    agg = aggregate(runs, topics)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps({"aggregate": agg, "runs": runs}, indent=1), encoding="utf-8")
    md = render_markdown(agg, args.label)
    (out / f"{args.label}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
