import pytest
from app.queue import claim_stale_jobs, consume_jobs, ensure_group, push_job, requeue_job, start_attempt


@pytest.fixture
def qconfig(config):
    config.stream_key = "research:jobs"
    config.consumer_group = "workers"
    config.consumer_name = "worker-new"
    config.job_reclaim_idle_ms = 0
    return config


async def _take(redis, cfg, consumer):
    cfg.consumer_name = consumer
    msgs = await redis.xreadgroup(cfg.consumer_group, consumer, {cfg.stream_key: ">"}, count=1)
    [(_, [(msg_id, data)])] = msgs
    # fakeredis returns stream entries as bytes even with decode_responses=True (real Redis returns str)
    text = lambda v: v.decode() if isinstance(v, bytes) else v
    return text(msg_id), {text(k): text(v) for k, v in data.items()}


async def test_job_left_by_dead_worker_is_reclaimed(redis, qconfig):
    await ensure_group(redis, qconfig)
    job_id = await push_job(redis, qconfig, "AI chips", "s1", "text")
    dead_msg, _ = await _take(redis, qconfig, "worker-dead")          # delivered, never acked

    qconfig.consumer_name = "worker-new"
    [job] = await consume_jobs(redis, qconfig)                        # stale job comes before new ones
    assert job["msg_id"] == dead_msg and job["data"]["job_id"] == job_id


async def test_recently_delivered_job_is_not_stolen(redis, qconfig):
    qconfig.job_reclaim_idle_ms = 15 * 60 * 1000
    await ensure_group(redis, qconfig)
    await push_job(redis, qconfig, "AI chips", "s1", "text")
    await _take(redis, qconfig, "worker-busy")
    qconfig.consumer_name = "worker-new"
    assert await claim_stale_jobs(redis, qconfig) == []


async def test_requeue_hands_job_to_next_worker_exactly_once(redis, qconfig):
    await ensure_group(redis, qconfig)
    job_id = await push_job(redis, qconfig, "AI chips", "s1", "text", fresh=True)
    msg_id, data = await _take(redis, qconfig, "worker-stopping")

    await requeue_job(redis, qconfig, msg_id, data)

    pending = await redis.xpending(qconfig.stream_key, qconfig.consumer_group)
    assert pending["pending"] == 0                                    # old delivery acked
    new_msg, new_data = await _take(redis, qconfig, "worker-new")
    assert new_msg != msg_id and new_data == data and new_data["job_id"] == job_id and new_data["fresh"] == "1"


async def test_start_attempt_counts_restarts(redis):
    assert [await start_attempt(redis, "job-1") for _ in range(3)] == [1, 2, 3]
    assert await start_attempt(redis, "job-2") == 1
    assert 0 < await redis.ttl("job_attempts:job-1") <= 86400
