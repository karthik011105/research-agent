import httpx
import pytest
import respx
from app import agents
from app.agents import _tz_call, rate_limit_wait

# Shape of TensorZero's 502 when every Groq model was rate-limited (taken from real gateway logs)
RATE_LIMITED = (
    'All variants failed with errors: groq_gpt_oss_120b: Error 429 Too Many Requests from groq client: '
    '{"error":{"message":"Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7424, Requested 5125. '
    'Please try again in 34.1175s.","type":"tokens","code":"rate_limit_exceeded"}} '
    'groq_qwen: Error 429 Too Many Requests from groq client: {"error":{"message":"... Please try again in 3.63s.",'
    '"code":"rate_limit_exceeded"}}'
)
OK = {"content": [{"type": "thought", "text": "hmm"}, {"type": "text", "text": "answer"}]}


def test_rate_limit_wait_picks_shortest_retry_hint():
    assert rate_limit_wait(RATE_LIMITED) == pytest.approx(4.63)
    assert rate_limit_wait('rate_limit_exceeded "Please try again in 2m30.5s"') == pytest.approx(151.5)
    assert rate_limit_wait("rate_limit_exceeded, no hint") == 20.0
    assert rate_limit_wait("model_not_found") is None


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_sleep(seconds):
        waited.append(seconds)

    monkeypatch.setattr(agents.asyncio, "sleep", fake_sleep)
    return waited


@respx.mock
async def test_tz_call_waits_out_rate_limit_then_succeeds(config, sleeps):
    route = respx.post("http://tensorzero.test/inference").mock(side_effect=[
        httpx.Response(502, text=RATE_LIMITED),
        httpx.Response(200, json=OK),
    ])
    assert await _tz_call(config, "research_plan", "hi") == "answer"
    assert route.call_count == 2
    assert sleeps == [pytest.approx(4.63)]


@respx.mock
async def test_tz_call_gives_up_after_rate_limit_retries(config, sleeps):
    respx.post("http://tensorzero.test/inference").mock(return_value=httpx.Response(502, text=RATE_LIMITED))
    with pytest.raises(httpx.HTTPStatusError):
        await _tz_call(config, "research_plan", "hi")
    assert len(sleeps) == config.llm_rate_limit_retries


@respx.mock
async def test_tz_call_reports_daily_limit_instead_of_waiting(config, sleeps):
    respx.post("http://tensorzero.test/inference").mock(
        return_value=httpx.Response(502, text='rate_limit_exceeded "Please try again in 14m2s"'))
    with pytest.raises(RuntimeError, match="daily limit"):
        await _tz_call(config, "research_plan", "hi")
    assert sleeps == []


@respx.mock
async def test_tz_call_other_errors_use_normal_retries(config, sleeps):
    config.llm_max_retries = 2
    route = respx.post("http://tensorzero.test/inference").mock(
        return_value=httpx.Response(502, text='{"error":"model_not_found"}'))
    with pytest.raises(httpx.HTTPStatusError):
        await _tz_call(config, "research_plan", "hi")
    assert route.call_count == 2
