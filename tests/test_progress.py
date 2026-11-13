import json
from app import agents, progress
from app.agents import build_graph, initial_state
from tests.test_agents import FakeLLM, fake_search  # noqa: F401  (fake_search is a fixture)


async def test_emit_without_a_job_does_nothing():
    await progress.emit("planning")                     # no emitter installed: must not raise


async def test_a_failing_emitter_never_breaks_the_job():
    async def broken(kind, data):
        raise ConnectionError("redis down")
    token = progress.use(broken)
    try:
        await progress.emit("planning")
    finally:
        progress._emitter.reset(token)


async def test_redis_progress_round_trip(redis):
    token = progress.use(progress.RedisProgress(redis, "job1"))
    try:
        await progress.emit("planned", sub_questions=[{"id": 1}])
        await progress.emit("gap", kind="news", queries=["q"])      # "kind" is allowed as event data
        await progress.emit("done", origin="pipeline")
    finally:
        progress._emitter.reset(token)
    events = await progress.read_events(redis, "job1", "0", 10)
    assert [e["type"] for _, e in events] == ["planned", "gap", "done"]
    assert events[1][1]["kind"] == "news" and 0 < await redis.ttl("progress:job1") <= 3600
    later = await progress.read_events(redis, "job1", events[0][0], 10)        # resume after the first event
    assert [e["type"] for _, e in later] == ["gap", "done"]


async def test_sse_stream_replays_keeps_alive_and_stops_at_the_final_event():
    batches = [[], [("1-0", {"type": "planned"})], [("2-0", {"type": "done", "origin": "pipeline"})]]
    seen_last_ids = []

    async def read(last_id, block_ms):
        seen_last_ids.append(last_id)
        return batches.pop(0)

    chunks = [c async for c in progress.sse_events(read, "job1", keepalive_s=0.01)]
    assert chunks[0] == ": keepalive\n\n"
    assert chunks[1].startswith("id: 1-0\nevent: progress\ndata: ") and json.loads(chunks[1].split("data: ")[1])["type"] == "planned"
    assert '"done"' in chunks[2] and len(chunks) == 3
    assert seen_last_ids == ["0", "0", "1-0"]                 # resumes from the last delivered event


async def test_pipeline_emits_a_timeline_including_parallel_branches(config, redis, monkeypatch, fake_search):
    events = []

    async def capture(kind, data):
        events.append((kind, data))

    monkeypatch.setattr(agents, "_tz_call", FakeLLM(critic_replies=["NO\n- fix X", "YES"], gap_reply={"queries": []}))
    token = progress.use(capture)
    try:
        await build_graph(config, redis).ainvoke(initial_state("LLM guardrails", "s", [], ""))
    finally:
        progress._emitter.reset(token)
    kinds = [k for k, _ in events]
    assert kinds[:2] == ["planning", "planned"]
    assert kinds.count("searching") == 2 and kinds.count("searched") == 2          # one per sub-question
    assert kinds.count("extracting") == 2 and kinds.count("extracted") == 2
    assert kinds.index("sources") < kinds.index("extracting") < kinds.index("writing")
    assert [d["draft"] for k, d in events if k == "writing"] == [1, 2]              # critic sent it back once
    checked = [d for k, d in events if k == "checked"]
    assert [c["passed"] for c in checked] == [False, True]
    planned = next(d for k, d in events if k == "planned")
    assert [sq["question"] for sq in planned["sub_questions"]] == ["What techniques exist?", "What do practitioners say?"]
