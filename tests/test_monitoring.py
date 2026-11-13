from datetime import datetime, timedelta
from app.monitoring import is_due, schedule_due, slot_key, what_changed

OLD = """## Executive Summary
The EU AI Act's obligations for general-purpose AI apply from August 2025 [1]. Fines can reach 7% of turnover [2].

## Key Findings
- The Commission published a code of practice for providers [1].
- Several member states have not yet named their regulators [3].
"""
NEW = """## Executive Summary
The EU AI Act's obligations for general-purpose AI apply from August 2025 [1]. Fines can reach 7% of global turnover [2].

## Key Findings
- The Commission published a code of practice for AI providers [1].
- The Digital Omnibus proposes delaying high-risk obligations to December 2027 [4].
"""


def test_what_changed_finds_new_and_dropped_claims_ignoring_rewording():
    changes = what_changed(OLD, NEW)
    assert changes["first_run"] is False
    assert changes["new"] == ["The Digital Omnibus proposes delaying high-risk obligations to December 2027."]
    assert changes["dropped"] == ["Several member states have not yet named their regulators."]
    assert changes["unchanged"] == 3                       # small rewordings ("global turnover") count as the same


def test_first_run_has_nothing_to_compare():
    assert what_changed(None, NEW) == {"first_run": True, "new": [], "dropped": [], "unchanged": 4}


def test_is_due():
    now = datetime(2026, 11, 14, 12, 0)
    m = {"id": "m1", "interval_hours": 24, "active": True, "last_run_at": None}
    assert is_due(m, now)
    assert not is_due({**m, "last_run_at": now - timedelta(hours=5)}, now)
    assert is_due({**m, "last_run_at": now - timedelta(hours=24)}, now)
    assert not is_due({**m, "active": False}, now)


class MonitorStore:
    def __init__(self, monitors):
        self.monitors = monitors

    async def list_monitors(self):
        return self.monitors

    async def mark_started(self, monitor_id, when, job_id):
        for m in self.monitors:
            if m["id"] == monitor_id:
                m["last_run_at"], m["last_job_id"] = when, job_id


async def test_scheduler_queues_each_due_monitor_once_even_with_two_tasks(redis):
    now = datetime(2026, 11, 14, 12, 0)
    monitors = [{"id": "m1", "topic": "EU AI Act", "interval_hours": 24, "active": True, "last_run_at": None},
                {"id": "m2", "topic": "EV sales", "interval_hours": 24, "active": True,
                 "last_run_at": now - timedelta(hours=2)}]
    queued_jobs = []

    async def enqueue(m):
        queued_jobs.append(m["topic"])
        return f"job-{m['id']}"

    task_a = MonitorStore([dict(m) for m in monitors])
    task_b = MonitorStore([dict(m) for m in monitors])        # a second app task with its own view
    assert await schedule_due(task_a, redis, enqueue, now) == ["m1"]
    assert await schedule_due(task_b, redis, enqueue, now) == []      # same slot: the Redis lock says no
    assert queued_jobs == ["EU AI Act"] and task_a.monitors[0]["last_job_id"] == "job-m1"
    assert slot_key(monitors[0], now) != slot_key(monitors[0], now + timedelta(hours=24))
