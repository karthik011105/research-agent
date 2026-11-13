import pytest
from app import config as config_module
from app.config import Config
from app.guardrails import validate_input, validate_output

LOCAL_ENV = {
    "CONFIG_SOURCE": "env",
    "REDIS_URL": "redis://redis:6379/0",
    "DATABASE_URL": "postgresql://research:research@postgres:5432/research",
    "TENSORZERO_URL": "http://tensorzero:3000",
    "API_KEY": "local-key",
}


@pytest.fixture
def local_env(monkeypatch):
    for k, v in LOCAL_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("BEDROCK_GUARDRAIL_VERSION", raising=False)
    config_module._load_secret.cache_clear()
    yield monkeypatch
    config_module._load_secret.cache_clear()


def test_env_config_does_not_touch_secrets_manager(local_env, monkeypatch):
    def no_aws(*args, **kwargs):
        raise AssertionError("local config must not call AWS")

    monkeypatch.setattr(config_module.boto3, "client", no_aws)
    local_env.setenv("GUARDRAILS_ENABLED", "false")
    local_env.setenv("SEARCH_MAX_SOURCES", "12")
    cfg = Config()
    assert cfg.database_url.endswith("@postgres:5432/research")
    assert cfg.tensorzero_url == "http://tensorzero:3000"
    assert cfg.search_max_sources == 12 and cfg.api_key == "local-key"
    assert cfg.guardrails_enabled is False and cfg.bedrock_guardrail_id == ""


def test_guardrails_default_on_and_require_their_ids(local_env):
    local_env.delenv("GUARDRAILS_ENABLED", raising=False)
    with pytest.raises(KeyError, match="BEDROCK_GUARDRAIL_ID"):
        Config()       # production secrets never set GUARDRAILS_ENABLED, so a missing id must fail loudly
    local_env.setenv("BEDROCK_GUARDRAIL_ID", "gr-123")
    local_env.setenv("BEDROCK_GUARDRAIL_VERSION", "2")
    config_module._load_secret.cache_clear()   # settings are read once per process
    cfg = Config()
    assert cfg.guardrails_enabled is True and cfg.bedrock_guardrail_version == "2"


async def test_disabled_guardrails_skip_bedrock(local_env, monkeypatch):
    local_env.setenv("GUARDRAILS_ENABLED", "false")
    cfg = Config()

    def no_bedrock(*args, **kwargs):
        raise AssertionError("guardrails are off: Bedrock must not be called")

    monkeypatch.setattr("app.guardrails._apply_guardrail_sync", no_bedrock)
    assert await validate_input(cfg, "anything") == (True, "")
    assert await validate_output(cfg, "anything") == (True, "", [])
