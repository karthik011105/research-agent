import asyncio
import logging
import boto3
from app.config import Config
from app.retry import with_retry

logger = logging.getLogger(__name__)


def _apply_guardrail_sync(config: Config, text: str, source: str) -> dict:
    client = boto3.client("bedrock-runtime", region_name=config.aws_region)
    return client.apply_guardrail(
        guardrailIdentifier=config.bedrock_guardrail_id,
        guardrailVersion=config.bedrock_guardrail_version,
        source=source,
        content=[{"text": {"text": text}}],
    )


def intervened_policies(response: dict) -> list[str]:
    """Which policies fired, e.g. ["filter:MISCONDUCT:LOW", "topic:weapons"] - for logs and rewrite hints."""
    fired = []
    for a in response.get("assessments", []):
        for t in a.get("topicPolicy", {}).get("topics", []):
            if t.get("action") == "BLOCKED":
                fired.append(f"topic:{t.get('name')}")
        for f in a.get("contentPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                fired.append(f"filter:{f.get('type')}:{f.get('confidence')}")
        words = a.get("wordPolicy", {})
        for w in words.get("customWords", []) + words.get("managedWordLists", []):
            if w.get("action") == "BLOCKED":
                fired.append("word")
        for p in a.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
            if p.get("action") == "BLOCKED":
                fired.append(f"pii:{p.get('type')}")
    return fired


async def validate_input(config: Config, text: str) -> tuple[bool, str]:
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "INPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        logger.warning(f"Input guardrail intervened: {intervened_policies(response)}")
        return False, "Input blocked by safety guardrail."
    return True, ""


async def validate_output(config: Config, text: str) -> tuple[bool, str, list[str]]:
    """(ok, user-facing reason, policies that fired)."""
    response = await with_retry(
        lambda: asyncio.to_thread(_apply_guardrail_sync, config, text, "OUTPUT"),
        max_retries=config.llm_max_retries,
        delay=config.llm_retry_delay,
    )
    if response.get("action") == "GUARDRAIL_INTERVENED":
        fired = intervened_policies(response)
        logger.warning(f"Output guardrail intervened: {fired}")
        return False, "Output blocked by safety guardrail.", fired
    return True, "", []
