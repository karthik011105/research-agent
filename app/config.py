import os
import socket
import boto3
import json
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_secret() -> dict:
    # Local development (docker-compose.yml): every setting comes from environment variables / .env,
    # so the app runs with the AWS stack paused. Production reads the Secrets Manager secret.
    if os.environ.get("CONFIG_SOURCE") == "env":
        return dict(os.environ)
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId="research-agent/config")
    return json.loads(response["SecretString"])


class Config:
    def __init__(self):
        data = _load_secret()

        # AWS
        self.aws_region: str = data.get("AWS_REGION", "us-east-1")

        # Bedrock Guardrails. Always on in production (the secret has no GUARDRAILS_ENABLED key); local runs may
        # turn them off when no AWS credentials are mounted into the container.
        self.guardrails_enabled: bool = data.get("GUARDRAILS_ENABLED", "true").strip().lower() != "false"
        self.bedrock_guardrail_id: str = data["BEDROCK_GUARDRAIL_ID"] if self.guardrails_enabled else ""
        # Task env wins: Terraform publishes a new version whenever the guardrail policy changes
        self.bedrock_guardrail_version: str = (
            (os.environ.get("BEDROCK_GUARDRAIL_VERSION") or data["BEDROCK_GUARDRAIL_VERSION"])
            if self.guardrails_enabled else ""
        )

        # Storage
        # Task env wins: Terraform sets it per deploy, while the secret's copy can go stale across pause/resume
        self.redis_url: str = os.environ.get("REDIS_URL") or data["REDIS_URL"]
        self.database_url: str = data["DATABASE_URL"]
        self.tensorzero_url: str = data["TENSORZERO_URL"]

        # Auth
        self.api_key: str = data.get("API_KEY", "")

        # LangSmith tracing
        self.langsmith_api_key: str = data.get("LANGSMITH_API_KEY", "")
        self.langchain_project: str = data.get("LANGCHAIN_PROJECT", "research-agent")
        self.langsmith_dataset: str = data.get("LANGSMITH_DATASET", "research-agent-reports")

        # Semantic cache
        self.cache_ttl: int = int(data.get("CACHE_TTL", 3600))
        self.cache_similarity_threshold: float = float(data.get("CACHE_SIMILARITY_THRESHOLD", 0.85))
        self.cache_max_entries: int = int(data.get("CACHE_MAX_ENTRIES", 1000))

        # Session memory
        self.session_ttl: int = int(data.get("SESSION_TTL", 1800))
        self.session_max_messages: int = int(data.get("SESSION_MAX_MESSAGES", 5))
        self.session_content_truncate: int = int(data.get("SESSION_CONTENT_TRUNCATE", 500))

        # Long-term memory
        # Reports older than this are re-researched with live search instead of reused
        self.ltm_fresh_hours: int = int(data.get("LTM_FRESH_HOURS", 24))
        self.ltm_threshold: float = float(data.get("LTM_THRESHOLD", 0.88))
        # Step 5c memory-poisoning defenses (measured before being turned on):
        #   MEMORY_SCOPE=session   reuse cached/remembered reports only within the session that produced them
        #   MEMORY_VERIFIED_ONLY   store a report only if the critic approved it (no open fix/remove items)
        self.memory_scope: str = data.get("MEMORY_SCOPE", "global").strip().lower()
        self.memory_verified_only: bool = data.get("MEMORY_VERIFIED_ONLY", "false").strip().lower() == "true"
        # Minimum similarity for a past report to be offered to the writer as background on a related topic
        self.ltm_related_min: float = float(data.get("LTM_RELATED_MIN", 0.65))
        self.ltm_diff_threshold: float = float(data.get("LTM_DIFF_THRESHOLD", 0.7))
        self.ltm_diff_limit: int = int(data.get("LTM_DIFF_LIMIT", 5))
        self.ivfflat_lists: int = int(data.get("IVFFLAT_LISTS", 100))

        # Job queue
        self.stream_key: str = data.get("STREAM_KEY", "research:jobs")
        self.consumer_group: str = data.get("CONSUMER_GROUP", "workers")
        # hostname = unique per ECS task = safe for horizontal scaling
        self.consumer_name: str = data.get("CONSUMER_NAME", socket.gethostname())
        self.result_ttl: int = int(data.get("RESULT_TTL", 3600))
        # A pending job untouched this long belonged to a worker that died; another worker takes it over.
        # Longer than the slowest real job (rate-limit waits included) so a live worker is never overtaken.
        self.job_reclaim_idle_ms: int = int(data.get("JOB_RECLAIM_IDLE_MS", 15 * 60 * 1000))
        self.job_max_attempts: int = int(data.get("JOB_MAX_ATTEMPTS", 3))

        # Live web search (Tavily primary, Brave optional fallback)
        self.tavily_api_key: str = data.get("TAVILY_API_KEY", "")
        self.brave_api_key: str = data.get("BRAVE_API_KEY", "")
        self.search_max_results: int = int(data.get("SEARCH_MAX_RESULTS", 5))
        self.search_max_sources: int = int(data.get("SEARCH_MAX_SOURCES", 8))
        self.search_max_queries: int = int(data.get("SEARCH_MAX_QUERIES", 4))
        self.search_max_rounds: int = int(data.get("SEARCH_MAX_ROUNDS", 2))
        self.search_daily_limit: int = int(data.get("SEARCH_DAILY_LIMIT", 40))
        self.search_cache_ttl: int = int(data.get("SEARCH_CACHE_TTL", 21600))
        self.source_char_limit: int = int(data.get("SOURCE_CHAR_LIMIT", 1500))

        # Multi-source research (Phase 1). Defaults live here because the secret's JSON is not managed by
        # Terraform after creation — add a key to research-agent/config only to override one.
        self.enabled_providers: list[str] = [
            p.strip() for p in data.get(
                "ENABLED_PROVIDERS", "tavily,brave,openalex,arxiv,semantic_scholar,gdelt,hackernews").split(",") if p.strip()
        ]
        self.semantic_scholar_api_key: str = data.get("SEMANTIC_SCHOLAR_API_KEY", "")
        # Sources never cited: prediction/betting markets state odds, not facts. Subdomains are included.
        self.blocked_domains: list[str] = [
            d.strip().lower() for d in data.get(
                "BLOCKED_DOMAINS", "polymarket.com,kalshi.com,manifold.markets,predictit.org").split(",") if d.strip()
        ]
        self.max_subquestions: int = int(data.get("MAX_SUBQUESTIONS", 3))
        self.queries_per_subquestion: int = int(data.get("QUERIES_PER_SUBQUESTION", 2))
        self.scrape_max_pages: int = int(data.get("SCRAPE_MAX_PAGES", 4))
        self.scrape_min_chars: int = int(data.get("SCRAPE_MIN_CHARS", 400))
        self.scrape_timeout: float = float(data.get("SCRAPE_TIMEOUT", 10))
        self.scrape_max_bytes: int = int(data.get("SCRAPE_MAX_BYTES", 1_000_000))
        # Parallel LLM calls per worker (extract branches); free-tier per-minute token limits are shared
        self.llm_concurrency: int = int(data.get("LLM_CONCURRENCY", 2))

        # Agent tuning
        self.agent_report_truncate: int = int(data.get("AGENT_REPORT_TRUNCATE", 3000))
        self.agent_max_iterations: int = int(data.get("AGENT_MAX_ITERATIONS", 2))
        self.critic_report_truncate: int = int(data.get("CRITIC_REPORT_TRUNCATE", 8000))
        self.critic_source_chars: int = int(data.get("CRITIC_SOURCE_CHARS", 600))
        # Step 3: "single" (one YES/NO critic), "panel" (fact-checker + skeptic) or "debate" (panel + debate on
        # disagreements). Stays "single" until the Step 3 comparison shows which setup is best.
        self.critic_mode: str = data.get("CRITIC_MODE", "single").strip().lower()
        self.critic_debate_rounds: int = int(data.get("CRITIC_DEBATE_ROUNDS", 2))
        # Step 4: use learned behaviour (writer bandit, critic trust, learned debate rounds). Off = rule-based
        # baseline; labelled critic evaluations still train the learners either way.
        self.learning_enabled: bool = data.get("LEARNING_ENABLED", "false").strip().lower() == "true"
        self.learning_explore: float = float(data.get("LEARNING_EXPLORE", 0.1))

        # Eval tuning
        self.eval_report_truncate: int = int(data.get("EVAL_REPORT_TRUNCATE", 1500))
        self.eval_comment_truncate: int = int(data.get("EVAL_COMMENT_TRUNCATE", 300))

        # LLM retry
        self.llm_max_retries: int = int(data.get("LLM_MAX_RETRIES", 3))
        self.llm_retry_delay: float = float(data.get("LLM_RETRY_DELAY", 1.0))
        # Free-tier LLMs hit per-minute token limits mid-pipeline; each retry waits as long as the provider asks
        self.llm_rate_limit_retries: int = int(data.get("LLM_RATE_LIMIT_RETRIES", 5))

        # Rate limiting (per IP, per window)
        self.rate_limit_requests: int = int(data.get("RATE_LIMIT_REQUESTS", 10))
        self.rate_limit_window: int = int(data.get("RATE_LIMIT_WINDOW", 60))

        # DB connection pool
        self.db_pool_min: int = int(data.get("DB_POOL_MIN", 2))
        self.db_pool_max: int = int(data.get("DB_POOL_MAX", 10))

        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
