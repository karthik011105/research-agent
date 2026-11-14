from app.history import history_record


def _result(**over):
    base = {"status": "done", "topic": "EV sales 2026", "origin": "pipeline", "live_sources": True,
            "sources_count": 2, "review_notes": ["one point"], "sub_questions": ["q1"],
            "sources": [{"id": 1, "title": "A", "url": "https://a.example"}],
            "report": "**Executive Summary**\nChina sold most EVs [1].\n\n## Key Findings\n- Point [1]\n\n## Sources\n[1] A"}
    return {**base, **over}


def test_a_finished_report_becomes_a_history_row():
    row = history_record("job1", "sess1", _result())
    assert row["job_id"] == "job1" and row["session_id"] == "sess1" and row["topic"] == "EV sales 2026"
    assert row["sources_count"] == 2 and row["notes_count"] == 1 and row["live_sources"] is True
    assert row["sources"][0]["url"] == "https://a.example" and row["sub_questions"] == ["q1"]
    # the preview is readable text: no headings, no source list
    assert row["preview"].startswith("China sold most EVs") and "## Sources" not in row["preview"]
    assert "Key Findings" not in row["preview"]


def test_instant_replies_errors_and_empty_reports_are_not_kept():
    assert history_record("j", "s", _result(origin="assistant")) is None
    assert history_record("j", "s", {"status": "error", "error": "boom"}) is None
    assert history_record("j", "s", _result(report="")) is None


def test_cached_answers_are_kept_and_previews_are_capped():
    row = history_record("j", "s", _result(origin="cache", report="word " * 400, review_notes=None, sources=None))
    assert row["origin"] == "cache" and row["notes_count"] == 0 and row["sources"] == []
    assert len(row["preview"]) <= 280


def test_the_web_app_pages_and_report_endpoints_are_routed():
    # read from the source: the test environment doesn't install FastAPI (see test_imports.py)
    import ast, pathlib
    src = (pathlib.Path(__file__).parent.parent / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    pages = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "WEB_PAGES")
    assert set(pages) == {"/", "/history", "/history/{job_id}", "/dashboard", "/watch"}
    for route in ('@app.get("/reports"', '@app.get("/reports/{job_id}"', '@app.get("/reports/{job_id}/pdf"'):
        assert route in src
    # no web-app page may shadow an API path
    api_paths = {d.args[0].value for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 for d in n.decorator_list if isinstance(d, ast.Call) and getattr(d.func, "attr", "") in ("get", "post", "delete")
                 and d.args and isinstance(d.args[0], ast.Constant)}
    assert not (set(pages) - {"/"}) & api_paths
