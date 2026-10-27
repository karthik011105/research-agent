import asyncio
import json
import re
import httpx
import logging
from datetime import date
from typing import TypedDict
import redis.asyncio as aioredis
from langgraph.graph import StateGraph, END, START
from langsmith import traceable
from app.config import Config
from app.retry import with_retry
from app.tools.search import SearchUnavailable, search_many, format_sources

logger = logging.getLogger(__name__)

_SEARCH_TOPICS = {"general", "news", "finance"}


class ResearchState(TypedDict):
    topic: str
    session_id: str
    session_history: list[dict]  # prior conversation turns passed into agent
    ltm_context: str             # related previous report passed to writer
    queries: list[str]           # queries for the next search round
    search_topic: str            # tavily topic: general | news | finance
    sources: list[dict]          # numbered web sources {id, title, url, content, ...}
    search_rounds: int
    needs_more: bool             # gap check asked for another search round
    live_sources: bool           # False = search unavailable, report is from model knowledge
    facts: str                   # cited facts extracted from sources
    report: str
    verified: bool
    critic_feedback: str
    error: str
    iterations: int


def initial_state(topic: str, session_id: str, session_history: list[dict], ltm_context: str) -> ResearchState:
    return ResearchState(
        topic=topic,
        session_id=session_id,
        session_history=session_history,
        ltm_context=ltm_context,
        queries=[],
        search_topic="general",
        sources=[],
        search_rounds=0,
        needs_more=False,
        live_sources=False,
        facts="",
        report="",
        verified=False,
        critic_feedback="",
        error="",
        iterations=0,
    )


_RETRY_IN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
_MAX_RATE_LIMIT_WAIT = 65.0  # per-minute limits reset within a minute; longer waits mean a daily limit


def rate_limit_wait(error_body: str) -> float | None:
    """Seconds to wait when every model was rate-limited, None if this isn't a rate-limit error.
    TensorZero's 502 body carries each provider's error; the shortest 'try again in' wins."""
    if "rate_limit" not in error_body and "429" not in error_body:
        return None
    waits = [int(m or 0) * 60 + float(s) for m, s in _RETRY_IN.findall(error_body)]
    return min(waits) + 1.0 if waits else 20.0


async def _tz_call(config: Config, function_name: str, message: str) -> str:
    """with_retry handles transient errors; a rate-limited call instead waits as long as the provider asks."""
    for attempt in range(config.llm_rate_limit_retries + 1):
        try:
            return await with_retry(
                lambda: _tz_call_once(config, function_name, message),
                max_retries=config.llm_max_retries,
                delay=config.llm_retry_delay,
                retry_if=lambda e: not _is_rate_limited(e),
            )
        except httpx.HTTPStatusError as e:
            wait = rate_limit_wait(e.response.text)
            if wait is None or attempt == config.llm_rate_limit_retries:
                raise
            if wait > _MAX_RATE_LIMIT_WAIT:
                raise RuntimeError(f"LLM daily limit reached for all models (retry in {wait / 60:.0f} min)") from e
            logger.warning(f"{function_name}: all models rate-limited, waiting {wait:.0f}s ({attempt + 1}/{config.llm_rate_limit_retries})")
            await asyncio.sleep(wait)


def _is_rate_limited(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and rate_limit_wait(exc.response.text) is not None


def response_text(body: dict) -> str:
    """Joins the answer's text blocks. Reasoning models may also return "thought" blocks, which are skipped."""
    text = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
    if not text.strip():
        raise ValueError(f"TensorZero returned no text content: {str(body)[:300]}")
    return text


async def _tz_call_once(config: Config, function_name: str, message: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{config.tensorzero_url}/inference",
            json={
                "function_name": function_name,
                "input": {"messages": [{"role": "user", "content": message}]},
            },
        )
        response.raise_for_status()
        return response_text(response.json())


def parse_json_object(text: str) -> dict:
    """Pull the first {...} block out of an LLM reply. Returns {} if there is none or it doesn't parse."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def clean_queries(raw, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    queries, seen = [], set()
    for q in raw:
        if isinstance(q, str) and q.strip() and q.strip().lower() not in seen:
            seen.add(q.strip().lower())
            queries.append(q.strip()[:200])
    return queries[:limit]


def cited_ids(report: str) -> set[int]:
    return {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", report) for n in re.split(r"\s*,\s*", group)}


def sources_section(report: str, sources: list[dict]) -> str:
    """Built from our own source list, not by the LLM, so URLs can't be invented."""
    used = cited_ids(report)
    lines = [f"[{s['id']}] {s['title']} — {s['url']}" for s in sources if s["id"] in used]
    return "\n\n## Sources\n" + "\n".join(lines) if lines else ""


def strip_citations(report: str) -> str:
    """A previous report's [n] markers point at *its* sources, not this run's — remove them and its Sources list."""
    report = re.split(r"\n#+\s*Sources\s*\n", report)[0]
    return re.sub(r"\s?\[\d+(?:\s*,\s*\d+)*\]", "", report).rstrip()


def _history_block(session_history: list[dict]) -> str:
    if not session_history:
        return ""
    recent = session_history[-4:]  # last 4 turns for context
    return (
        "\n\nPrevious conversation context (use this to understand what the user already knows and what angle they care about):\n"
        + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    )


class PlannerAgent:
    """Turns the topic into a handful of web search queries. Knows today's date so queries target current info."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:planner")
    async def run(self, topic: str, session_history: list[dict]) -> tuple[list[str], str]:
        logger.info(f"PlannerAgent: planning searches for '{topic}'")
        reply = await _tz_call(
            self.config,
            "research_plan",
            f"Today's date: {date.today().isoformat()}\n"
            f"Research topic: {topic}"
            f"{_history_block(session_history)}\n\n"
            f"Write up to {self.config.search_max_queries} web search queries that together cover this topic "
            f"(current state, key numbers, recent developments, main players or viewpoints). "
            f'Set "search_topic" to "news" if the topic is mainly about recent events, '
            f'"finance" if it is about markets or companies\' financials, otherwise "general".\n'
            f'Reply with JSON only: {{"search_topic": "general", "queries": ["...", "..."]}}',
        )
        parsed = parse_json_object(reply)
        queries = clean_queries(parsed.get("queries"), self.config.search_max_queries) or [topic]
        search_topic = parsed.get("search_topic") if parsed.get("search_topic") in _SEARCH_TOPICS else "general"
        return queries, search_topic


class ExtractAgent:
    """Pulls cited facts out of the web sources. Falls back to model knowledge when there are no sources."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:extract")
    async def run(self, topic: str, sources: list[dict]) -> str:
        if not sources:
            logger.info("ExtractAgent: no web sources, using model knowledge")
            return await _tz_call(
                self.config,
                "research_summarize",
                f"You are a research specialist. Find and list 5 key facts, recent developments, "
                f"and important details about: {topic}. Be thorough and specific.",
            )
        logger.info(f"ExtractAgent: extracting facts from {len(sources)} sources")
        return await _tz_call(
            self.config,
            "research_extract",
            f"Topic: {topic}\n"
            f"Today's date: {date.today().isoformat()}\n\n"
            f"Extract the key facts about the topic from the sources below as bullet points. "
            f"End every bullet with the id of the source it came from, e.g. [2] or [1, 3].\n\n"
            f"{format_sources(sources)}",
        )


class GapCheckAgent:
    """Decides whether the extracted facts leave an important gap worth one more search round."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:gap_check")
    async def run(self, topic: str, facts: str, previous_queries: list[str]) -> list[str]:
        reply = await _tz_call(
            self.config,
            "research_plan",
            f"Today's date: {date.today().isoformat()}\n"
            f"Research topic: {topic}\n"
            f"Queries already searched: {json.dumps(previous_queries)}\n\n"
            f"Facts found so far:\n{facts}\n\n"
            f"Is an important part of the topic still missing? If yes, give at most 2 new search queries "
            f"that would fill the gap. If the facts are enough, return an empty list.\n"
            f'Reply with JSON only: {{"queries": []}}',
        )
        return clean_queries(parse_json_object(reply).get("queries"), 2)


class WriterAgent:
    """
    Produces the final structured report from cited facts. If a related previous report exists in LTM,
    it uses it as reference so the new report builds on existing knowledge rather than starting from scratch.
    """

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:writer")
    async def run(self, topic: str, facts: str, live_sources: bool, ltm_context: str, critic_feedback: str) -> str:
        logger.info("WriterAgent: drafting report")

        ltm_section = ""
        if ltm_context:
            ltm_section = (
                f"\n\nPREVIOUS RESEARCH ON A RELATED TOPIC (use this as reference — "
                f"build on it, correct outdated information, and highlight what has changed. "
                f"It has no citations: anything you take from it must be confirmed by the current findings to be cited):\n"
                f"{strip_citations(ltm_context)[:2000]}"
            )
        feedback_section = ""
        if critic_feedback:
            feedback_section = f"\n\nA reviewer rejected the previous draft. Fix these problems:\n{critic_feedback}"
        citation_rule = (
            "Keep the [n] source citations from the findings on every claim that uses them. "
            "Do not add a sources or references list — it is appended automatically."
            if live_sources else
            "Live web search was unavailable, so these findings come from model knowledge. "
            "Say so briefly in the Executive Summary and do not invent citations."
        )

        return await _tz_call(
            self.config,
            "report_write",
            f"Today's date: {date.today().isoformat()}\n"
            f"Write a comprehensive, well-structured research report on: '{topic}'\n\n"
            f"Current research findings:\n{facts}"
            f"{ltm_section}{feedback_section}\n\n"
            f"Include: Executive Summary, Key Findings, Analysis, and Conclusion.\n"
            f"{citation_rule}",
        )


class CriticAgent:
    """Checks the report's claims against the cited sources (or for basic coherence when there are none)."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:critic")
    async def run(self, report: str, sources: list[dict]) -> tuple[bool, str]:
        logger.info("CriticAgent: verifying report")
        if sources:
            # Only the cited sources, shortened — keeps this call inside free-tier per-minute token limits
            used = cited_ids(report)
            cited = [{**s, "content": s["content"][:self.config.critic_source_chars]} for s in sources if s["id"] in used]
            prompt = (
                f"Check that the claims in this report are supported by the sources they cite, and that "
                f"no important claim is uncited or contradicts the sources.\n"
                f"Reply with YES if it passes, or NO followed by a short list of the problems.\n\n"
                f"REPORT:\n{report[:self.config.critic_report_truncate]}\n\n"
                f"SOURCES:\n{format_sources(cited) or '(the report cites no sources)'}"
            )
        else:
            prompt = (
                f"Review this report for factual consistency and logical coherence. "
                f"Reply with YES if it passes or NO with a brief reason:\n\n"
                f"{report[:self.config.agent_report_truncate]}"
            )
        check = await _tz_call(self.config, "research_verify", prompt)
        passed = check.strip().upper().startswith("YES")
        return passed, "" if passed else check.strip()[:1000]


class OrchestratorAgent:
    """
    Coordinates all sub-agents: plan → search → extract → (gap check → search again, at most
    search_max_rounds rounds) → write → verify. If the critic rejects the report it goes back to the
    writer with the critic's feedback (up to agent_max_iterations drafts) — no extra searches.
    """

    def __init__(self, config: Config, redis: aioredis.Redis):
        self.config = config
        self.redis = redis
        self.planner = PlannerAgent(config)
        self.extract_agent = ExtractAgent(config)
        self.gap_agent = GapCheckAgent(config)
        self.writer_agent = WriterAgent(config)
        self.critic_agent = CriticAgent(config)

    @traceable(run_type="chain", name="orchestrator:plan")
    async def plan_node(self, state: ResearchState) -> dict:
        queries, search_topic = await self.planner.run(state["topic"], state.get("session_history", []))
        return {"queries": queries, "search_topic": search_topic}

    @traceable(run_type="chain", name="orchestrator:search")
    async def search_node(self, state: ResearchState) -> dict:
        try:
            sources = await search_many(self.redis, self.config, state["queries"], state["sources"], state["search_topic"])
        except SearchUnavailable as e:
            logger.warning(f"Web search unavailable: {e}")
            return {"search_rounds": state["search_rounds"] + 1, "error": f"search: {e}"}
        return {
            "sources": sources,
            "live_sources": bool(sources),
            "search_rounds": state["search_rounds"] + 1,
        }

    @traceable(run_type="chain", name="orchestrator:extract")
    async def extract_node(self, state: ResearchState) -> dict:
        facts = await self.extract_agent.run(state["topic"], state["sources"])
        return {"facts": facts}

    @traceable(run_type="chain", name="orchestrator:gap_check")
    async def gap_check_node(self, state: ResearchState) -> dict:
        if not state["live_sources"] or state["search_rounds"] >= self.config.search_max_rounds:
            return {"needs_more": False}
        new_queries = await self.gap_agent.run(state["topic"], state["facts"], state["queries"])
        if new_queries:
            logger.info(f"Gap check: searching again with {new_queries}")
        return {"needs_more": bool(new_queries), "queries": new_queries or state["queries"]}

    @traceable(run_type="chain", name="orchestrator:write")
    async def write_node(self, state: ResearchState) -> dict:
        report = await self.writer_agent.run(
            state["topic"],
            state["facts"],
            state["live_sources"],
            state.get("ltm_context", ""),
            state.get("critic_feedback", ""),
        )
        if state["live_sources"]:
            report += sources_section(report, state["sources"])
        return {"report": report, "iterations": state.get("iterations", 0) + 1}

    @traceable(run_type="chain", name="orchestrator:verify")
    async def verify_node(self, state: ResearchState) -> dict:
        verified, feedback = await self.critic_agent.run(state["report"], state["sources"])
        return {"verified": verified, "critic_feedback": feedback}

    def route_after_gap_check(self, state: ResearchState) -> str:
        return "search" if state["needs_more"] else "write"

    def route_after_verify(self, state: ResearchState) -> str:
        """Orchestrator decision: rewrite with critic feedback or finish."""
        if not state["verified"] and state.get("iterations", 0) < self.config.agent_max_iterations:
            logger.info(f"Critic rejected report — rewriting (iteration {state['iterations']})")
            return "write"
        return END


def build_graph(config: Config, redis: aioredis.Redis):
    orchestrator = OrchestratorAgent(config, redis)
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", orchestrator.plan_node)
    workflow.add_node("search", orchestrator.search_node)
    workflow.add_node("extract", orchestrator.extract_node)
    workflow.add_node("gap_check", orchestrator.gap_check_node)
    workflow.add_node("write", orchestrator.write_node)
    workflow.add_node("verify", orchestrator.verify_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "gap_check")
    workflow.add_conditional_edges(
        "gap_check",
        orchestrator.route_after_gap_check,
        {"search": "search", "write": "write"},
    )
    workflow.add_edge("write", "verify")
    workflow.add_conditional_edges(
        "verify",
        orchestrator.route_after_verify,
        {"write": "write", END: END},
    )

    return workflow.compile()
