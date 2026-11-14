import asyncio
import json
import operator
import re
import httpx
import logging
from datetime import date
from typing import Annotated, TypedDict
import redis.asyncio as aioredis
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langsmith import traceable
from app import metering, progress
from app.config import Config
from app.retry import with_retry
from app.tools.scrape import enrich_sources
from app.tools.search import (
    BUDGETED_KINDS, KINDS, RECENCY_DAYS, dedupe_new, drop_blocked, format_sources, interleave, merge_sources,
    prefer_recent, search_kind,
)

logger = logging.getLogger(__name__)

_GAP_QUERIES = 2          # web-search queries a gap-check round may spend
_GAP_RESERVE = 3          # source slots the first round leaves free, so a gap round can still add something
_RECENCY_WORDS = re.compile(
    r"\b(today|tonight|yesterday|this week|past week|last week|breaking|right now|this month|past month|"
    r"latest|recent|recently|newest|currently|so far this year)\b", re.IGNORECASE)
STRICT_FUNCTIONS = {"research_summarize", "report_write", "research_plan", "research_extract", "research_verify",
                    "critic_fact"}

ABOUT_ME = (
    "I'm the Autonomous Research Agent. Give me a topic and I research it for you: I split it into "
    "sub-questions, search the live web, news, research papers (OpenAlex, arXiv) and Hacker News, read the "
    "sources, and write a structured report where every claim is cited [n] to its source. A critic agent then "
    "checks the claims against those sources.\n\n"
    "Under the hood: a LangGraph multi-agent pipeline on AWS, using open-source models (gpt-oss, Qwen) served "
    "by Groq - not ChatGPT or the OpenAI API.\n\n"
    "Try a topic like \"latest developments in small language models on phones\" and tick Fresh search for "
    "today's sources."
)

_SMALL_TALK = re.compile(
    r"^(hi+|hello+|hey+|yo|hola|namaste|good (morning|afternoon|evening)|thanks?( you)?|thank u|ok(ay)?|help|"
    r"who (are|r) (you|u)|what (are|r) (you|u)|what can (you|u) do|how do (you|u) work|what is this( app)?|"
    r"introduce yourself)( there)?[\s!.?]*$",
    re.IGNORECASE,
)


def quick_reply(text: str) -> str | None:
    """Greetings and questions about the assistant itself get an instant answer: no LLM call, no searches."""
    if _SMALL_TALK.match(" ".join(text.split())):
        if re.match(r"\s*(thanks?|thank u)", text, re.IGNORECASE):
            return "You're welcome! Send another topic whenever you like."
        return ABOUT_ME
    return None


class SubQuestion(TypedDict):
    id: int
    question: str
    queries: list[str]
    kinds: list[str]      # web | news | paper | community


class ResearchState(TypedDict):
    topic: str
    session_id: str
    session_history: list[dict]  # prior conversation turns passed into agent
    ltm_context: str             # related previous report passed to writer
    direct_reply: str            # planner decided the input is not a research topic and answered it
    sub_questions: list[SubQuestion]  # every sub-question so far (planner + gap rounds)
    recency: str                 # "week" | "month" | "year" when the topic asks for recent news, else ""
    round_subq_ids: list[int]    # sub-questions being researched in the current round
    raw_results: Annotated[list[dict], operator.add]   # search branch outputs, tagged with round + sub-question
    branch_facts: Annotated[list[dict], operator.add]  # extract branch outputs, one per sub-question
    sources: list[dict]          # numbered sources {id, title, url, content, provider, kind, subq, ...}
    search_rounds: int
    scraped_count: int
    needs_more: bool             # gap check asked for another search round
    live_sources: bool           # False = search unavailable, report is from model knowledge
    facts: str                   # cited facts, grouped by sub-question
    report: str
    verified: bool
    critic_feedback: str
    claim_verdicts: list[dict]   # critic panel: per-claim verdicts of the latest check (empty in "single" mode)
    writer_variant: str          # Step 4c: TensorZero writer variant the bandit chose for this job ("" = default)
    error: str
    iterations: int


class SearchTask(TypedDict):
    round: int
    subq: SubQuestion
    calls: list[tuple[str, str, bool]]   # (query, kind, may spend paid web-search budget)
    recency: str                         # date limit for the searches ("" = none)


class ExtractTask(TypedDict):
    topic: str
    subq: SubQuestion
    sources: list[dict]


def initial_state(topic: str, session_id: str, session_history: list[dict], ltm_context: str) -> ResearchState:
    return ResearchState(
        topic=topic,
        session_id=session_id,
        session_history=session_history,
        ltm_context=ltm_context,
        direct_reply="",
        sub_questions=[],
        recency="",
        round_subq_ids=[],
        raw_results=[],
        branch_facts=[],
        sources=[],
        search_rounds=0,
        scraped_count=0,
        needs_more=False,
        live_sources=False,
        facts="",
        report="",
        verified=False,
        critic_feedback="",
        claim_verdicts=[],
        writer_variant="",
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


async def _tz_call(config: Config, function_name: str, message: str, variant: str | None = None) -> str:
    """with_retry handles transient errors; a rate-limited call instead waits as long as the provider asks.
    `variant` pins a TensorZero variant (the Step 4c writer bandit's choice)."""
    for attempt in range(config.llm_rate_limit_retries + 1):
        try:
            return await with_retry(
                lambda: _tz_call_once(config, function_name, message, variant),
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


def served_models(body: dict) -> list[str]:
    """Model names from TensorZero's raw provider responses (requested with include_raw_response)."""
    names = []
    for entry in body.get("raw_response") or []:
        try:
            names.append(json.loads(entry.get("data") or "{}").get("model") or "")
        except (ValueError, AttributeError):
            continue
    return [n for n in names if n]


async def _tz_call_once(config: Config, function_name: str, message: str, variant: str | None = None) -> str:
    payload = {
        "function_name": function_name,
        "input": {"messages": [{"role": "user", "content": message}]},
    }
    if variant:
        payload["variant_name"] = variant
    if metering.active():
        # Benchmark runs only: learn which model actually answered, and pin the no-fallback writer variant so every
        # benchmarked report comes from the same model (see research_model_strict in tensorzero.toml)
        payload["include_raw_response"] = True
        if function_name in STRICT_FUNCTIONS:
            payload["variant_name"] = "strict"
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{config.tensorzero_url}/inference", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage") or {}
        metering.record(llm_calls=1, input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0))
        for name in served_models(body)[-1:]:      # the last raw response is the one that succeeded
            metering.record_model(name)
        return response_text(body)


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


def clean_kinds(raw) -> list[str]:
    kinds = []
    for k in raw if isinstance(raw, list) else [raw]:
        if isinstance(k, str) and k.strip().lower() in KINDS and k.strip().lower() not in kinds:
            kinds.append(k.strip().lower())
    return kinds[:2] or ["web"]


def parse_plan(reply: str, topic: str, max_subquestions: int, max_queries: int) -> list[SubQuestion]:
    """Planner JSON -> validated sub-questions. Accepts the older flat {"queries": [...]} shape, and falls back
    to searching the topic itself, so a malformed plan never stops the pipeline."""
    parsed = parse_json_object(reply)
    plan = []
    items = parsed.get("sub_questions")
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question = item["question"].strip()[:300] if isinstance(item.get("question"), str) else ""
        queries = clean_queries(item.get("queries"), max_queries) or ([question[:200]] if question else [])
        if queries:
            plan.append({"question": question or queries[0], "queries": queries, "kinds": clean_kinds(item.get("kinds"))})
    if not plan:
        legacy = clean_queries(parsed.get("queries"), max_queries * max_subquestions)
        plan = [{"question": topic, "queries": legacy or [topic[:200]], "kinds": ["web"]}]
    return [SubQuestion(id=i, **sq) for i, sq in enumerate(plan[:max_subquestions], start=1)]


def parse_recency(reply: str, topic: str) -> str:
    """The planner's time window for the topic. If its reply has none, words such as "today" or "latest" in the
    topic still make the search prefer recent results."""
    value = parse_json_object(reply).get("recency")
    if isinstance(value, str) and value.strip().lower() in RECENCY_DAYS:
        return value.strip().lower()
    if value is None and _RECENCY_WORDS.search(topic):
        return "week" if re.search(r"\b(today|tonight|yesterday|week|breaking|right now)\b", topic, re.I) else "month"
    return ""


def plan_calls(sub_questions: list[SubQuestion], web_limit: int) -> dict[int, list[tuple[str, str, bool]]]:
    """Every (query, kind) pair to run, per sub-question. Paid web search (web/news kinds) is shared out
    round-robin - each sub-question's first query before anyone's second - until web_limit is spent.
    Past the limit, news still uses the free GDELT feed; plain web calls are dropped."""
    calls: dict[int, list[tuple[str, str, bool]]] = {sq["id"]: [] for sq in sub_questions}
    budget = web_limit
    for qi in range(max((len(sq["queries"]) for sq in sub_questions), default=0)):
        for sq in sub_questions:
            if qi >= len(sq["queries"]):
                continue
            for kind in sq["kinds"]:
                paid = kind in BUDGETED_KINDS and budget > 0
                if paid:
                    budget -= 1
                if paid or kind not in BUDGETED_KINDS or kind == "news":
                    calls[sq["id"]].append((sq["queries"][qi], kind, paid))
    return calls


def combine_facts(branch_facts: list[dict]) -> str:
    return "\n\n".join(f"### {b['question']}\n{b['facts'].strip()}" for b in sorted(branch_facts, key=lambda b: b["subq"]))


# 【1】 / 【1, 3】 / 【2†L4-L9】 (gpt-oss), ［1］ (full-width), [1†source] -> [1] / [1, 3] / [2]
_CITATION_VARIANTS = re.compile(r"[【［\[]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*(?:†[^】］\]]*)?[】］\]]")


def normalize_citations(text: str) -> str:
    """Rewrites the citation styles models produce into the [n] form the rest of the pipeline reads."""
    return _CITATION_VARIANTS.sub(lambda m: "[" + ", ".join(re.findall(r"\d+", m.group(1))) + "]", text)


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


REVIEW_NOTES_HEADING = "## Fact-check notes"


def review_notes(state: dict) -> list[str]:
    """What the critic still flagged in the final draft - empty when it approved it. Rewrites are capped, so a
    report can end with problems left; they are shown to the reader instead of being dropped silently."""
    if state.get("verified"):
        return []
    flagged = [c for c in state.get("claim_verdicts") or [] if c.get("decision") != "keep"]
    if flagged:
        return [(c.get("text") or "")[:240] + (f" ({c['fix'][:200]})" if c.get("fix") else "") for c in flagged][:10]
    text = re.sub(r"^\s*NO\b[\s.:\-\u2014]*", "", state.get("critic_feedback") or "", flags=re.IGNORECASE)
    lines = [line.strip().lstrip("-*\u2022 ").strip() for line in text.splitlines()]
    notes = [line[:300] for line in lines if len(line) > 3][:10]
    return notes or ["The fact-checker did not approve the final draft."]


def with_review_notes(report: str, notes: list[str]) -> str:
    """Appends the unresolved points as the report's last section, so the PDF, cache and memory keep them too."""
    if not notes:
        return report
    return (f"{report}\n\n{REVIEW_NOTES_HEADING}\nThe fact-checker still flagged these points after the final "
            f"rewrite. Check them against the cited sources before relying on them:\n"
            + "\n".join(f"- {n}" for n in notes))


def _history_block(session_history: list[dict]) -> str:
    if not session_history:
        return ""
    recent = session_history[-4:]  # last 4 turns for context
    return (
        "\n\nPrevious conversation context (use this to understand what the user already knows and what angle they care about):\n"
        + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    )


class PlannerAgent:
    """Splits the topic into sub-questions, writes search queries for each, and picks which kinds of source
    answer each one best. Knows today's date so queries target current information."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:planner")
    async def run(self, topic: str, session_history: list[dict]) -> tuple[list[SubQuestion], str, str]:
        """(plan, "", recency) for a research topic, or ([], answer, "") when the input is not something to research."""
        logger.info(f"PlannerAgent: planning research for '{topic}'")
        reply = await _tz_call(
            self.config,
            "research_plan",
            f"Today's date: {date.today().isoformat()}\n"
            f"Research topic: {topic}"
            f"{_history_block(session_history)}\n\n"
            f"Break the topic into at most {self.config.max_subquestions} sub-questions that together cover it "
            f"(current state, key numbers, recent developments, main players or viewpoints). For each one write up to "
            f"{self.config.queries_per_subquestion} short search queries and choose the one or two source kinds that "
            f"answer it best:\n"
            f'- "web": general web pages (the default)\n'
            f'- "news": news coverage of events from the last weeks or months\n'
            f'- "paper": academic papers (arXiv, Semantic Scholar) for scientific or technical research; '
            f"write these queries as 2-5 keywords\n"
            f'- "community": practitioner discussion on Hacker News, for opinions, developer experience and tools\n'
            f'Also set "recency" to how recent the information must be: "week" if the topic asks about today, '
            f'this week or breaking news; "month" if it asks for the latest, new or recent developments or this '
            f'month; "year" for this year; "" if it does not depend on dates.\n'
            f'Reply with JSON only: {{"recency": "", "sub_questions": [{{"question": "...", "queries": ["..."], '
            f'"kinds": ["web"]}}]}}\n\n'
            f"Exception: if the input is clearly not something to research - small talk, a question about you, or a "
            f"trivial fact or calculation - do not plan; reply with JSON only: "
            f'{{"intent": "chat", "reply": "<a short, friendly answer>"}}. A short subject such as "quantum computing" '
            f"IS a research topic. About you, if asked: {ABOUT_ME}",
        )
        parsed = parse_json_object(reply)
        if parsed.get("intent") == "chat" and isinstance(parsed.get("reply"), str) and parsed["reply"].strip():
            logger.info("PlannerAgent: not a research topic, answering directly")
            return [], parsed["reply"].strip()[:2000], ""
        plan = parse_plan(reply, topic, self.config.max_subquestions, self.config.queries_per_subquestion)
        recency = parse_recency(reply, topic)
        logger.info(f"PlannerAgent: {[(sq['question'], sq['kinds']) for sq in plan]}, recency={recency or 'any'}")
        return plan, "", recency


class ExtractAgent:
    """Pulls cited facts out of the sources for one sub-question. Falls back to model knowledge when there are no sources."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:extract")
    async def run(self, topic: str, sources: list[dict], question: str = "") -> str:
        if not sources:
            logger.info("ExtractAgent: no web sources, using model knowledge")
            return await _tz_call(
                self.config,
                "research_summarize",
                f"You are a research specialist. Find and list 5 key facts, recent developments, "
                f"and important details about: {topic}. Be thorough and specific.",
            )
        logger.info(f"ExtractAgent: extracting facts from {len(sources)} sources for '{question or topic}'")
        focus = f"Sub-question: {question}\n" if question and question != topic else ""
        return normalize_citations(await _tz_call(
            self.config,
            "research_extract",
            f"Topic: {topic}\n{focus}"
            f"Today's date: {date.today().isoformat()}\n\n"
            f"Extract the key facts that answer the {'sub-question' if focus else 'topic'} from the sources below "
            f"as at most 8 bullet points. Give the date of each event or announcement when the source states "
            f"it. Only include facts that answer the {'sub-question' if focus else 'topic'}. "
            f"End every bullet with the id of the source it came from, e.g. [2] or [1, 3].\n\n"
            f"{format_sources(sources)}",
        ))


class GapCheckAgent:
    """Decides whether the extracted facts leave an important gap worth one more search round."""

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:gap_check")
    async def run(self, topic: str, facts: str, previous_queries: list[str]) -> tuple[list[str], list[str]]:
        reply = await _tz_call(
            self.config,
            "research_plan",
            f"Today's date: {date.today().isoformat()}\n"
            f"Research topic: {topic}\n"
            f"Queries already searched: {json.dumps(previous_queries)}\n\n"
            f"Facts found so far:\n{facts}\n\n"
            f"Is an important part of the topic, as the user asked it, still missing? If yes, give at most "
            f"{_GAP_QUERIES} new search queries that would fill the gap and the source kind to search "
            f'("web", "news", "paper" or "community"). Stay within the question: do not widen it to related '
            f"subjects (such as regulation, ethics or controversies) unless the topic asks about them. "
            f"If the facts are enough, return an empty list.\n"
            f'Reply with JSON only: {{"queries": [], "kind": "web"}}',
        )
        parsed = parse_json_object(reply)
        return clean_queries(parsed.get("queries"), _GAP_QUERIES), clean_kinds(parsed.get("kind"))[:1]


class WriterAgent:
    """
    Produces the final structured report from cited facts. If a related previous report exists in LTM,
    it uses it as reference so the new report builds on existing knowledge rather than starting from scratch.
    """

    def __init__(self, config: Config):
        self.config = config

    @traceable(run_type="llm", name="agent:writer")
    async def run(self, topic: str, facts: str, live_sources: bool, ltm_context: str, critic_feedback: str,
                  variant: str | None = None, recency: str = "") -> str:
        logger.info("WriterAgent: drafting report")
        recency_rule = (
            "The topic asks for recent information: lead with the newest findings, give the date of each "
            "development, and say plainly when something is older (for example 'in March 2026'). Do not present "
            "older material as new.\n" if recency else "")

        ltm_section = ""
        if ltm_context:
            ltm_section = (
                f"\n\nBACKGROUND FROM EARLIER RESEARCH (may be on a different subject). Use it only where it is "
                f"about this exact topic, to note what has changed; otherwise ignore it. Never mention it or compare "
                f"against 'a previous report' — the reader has not seen it. It has no citations: anything you take "
                f"from it must be confirmed by the current findings to be cited):\n"
                f"{strip_citations(ltm_context)[:2000]}"
            )
        feedback_section = ""
        if critic_feedback:
            feedback_section = (f"\n\nReviewers checked the previous draft against its sources. Apply every item "
                                f"below (FIX = correct it to match the source, REMOVE = delete the claim), and keep "
                                f"everything else:\n{critic_feedback}")
        citation_rule = (
            "Keep the [n] source citations from the findings on every claim that uses them, written with plain "
            "square brackets exactly like [3] or [1, 4]. "
            "Do not add a sources or references list — it is appended automatically."
            if live_sources else
            "Live web search was unavailable, so these findings come from model knowledge. "
            "Say so briefly in the Executive Summary and do not invent citations."
        )

        return normalize_citations(await _tz_call(
            self.config,
            "report_write",
            f"Today's date: {date.today().isoformat()}\n"
            f"Write a comprehensive, well-structured research report on: '{topic}'\n\n"
            f"Current research findings (grouped by sub-question):\n{facts}"
            f"{ltm_section}{feedback_section}\n\n"
            f"Include: Executive Summary, Key Findings, Analysis, and Conclusion.\n"
            f"Cover every sub-question in the findings (each one under '###'); if one has no findings, say in one "
            f"line that no reliable information was found for it. Attribute each claim to what the source actually "
            f"says - do not merge details from different sources or documents into one claim, and never put "
            f"figures that belong to different models, products or organisations into one sentence.\n"
            f"{recency_rule}{citation_rule}",
            **({"variant": variant} if variant else {}),   # only when the writer bandit chose one
        ))


async def rewrite_for_safety(config: Config, state: dict, policies: list[str]) -> str:
    """One more draft after the output guardrail blocked the report: same findings, attacks and risks kept at
    a descriptive level. Used so a legitimate topic (e.g. defending against attacks) isn't lost after minutes
    of research because one passage read like instructions."""
    feedback = (
        f"An automated safety filter blocked the previous draft ({', '.join(policies) or 'policy match'}). "
        "Rewrite it for a general professional audience: focus on findings, defenses, impact and recommendations; "
        "describe attacks, exploits or misuse only at a high level (what they are and why they matter), with no "
        "step-by-step instructions, payloads, or operational detail."
    )
    report = await WriterAgent(config).run(state["topic"], state["facts"], state["live_sources"],
                                          state.get("ltm_context", ""), feedback, recency=state.get("recency", ""))
    return report + (sources_section(report, state["sources"]) if state["live_sources"] else "")


# The fact-checker gave no verdict (a reasoning model can spend its whole output budget thinking). The report is
# published with this note instead of failing the job; another rewrite wouldn't help, since nothing was checked.
CRITIC_UNAVAILABLE = "NO\n- The automated fact-check could not complete for this draft, so its claims were not checked."


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
        try:
            check = await _tz_call(self.config, "research_verify", prompt)
        except ValueError as e:
            if "no text content" not in str(e):
                raise
            logger.warning(f"CriticAgent: no verdict from the model, report left unverified ({str(e)[:120]})")
            return False, CRITIC_UNAVAILABLE
        passed = check.strip().upper().startswith("YES")
        return passed, "" if passed else check.strip()[:1000]


class OrchestratorAgent:
    """
    Coordinates the sub-agents as a map-reduce graph:

      plan -> search_branch x N -> merge -> extract_branch x N -> collect -> gap_check -> write -> verify
              (one per sub-question, (dedupe, global  (one per sub-question,          |  more?
               providers in parallel) ids, scrape)     LLM, concurrency-limited)      +-> search_branch (gap round)

    Sources are numbered once, in merge, so [n] citations from different branches never clash. The gap check
    can add one sub-question per round (at most search_max_rounds rounds). If the critic rejects the report it
    goes back to the writer with the critic's feedback (up to agent_max_iterations drafts) - no extra searches.
    """

    def __init__(self, config: Config, redis: aioredis.Redis, search_fn=None, learning=None, rulebook=None):
        self.config = config
        self.redis = redis
        # The benchmark injects a snapshot-replaying search; normal jobs use live search_kind
        self.search_fn = search_fn
        # Step 4 LearningService (None in tests and benchmark runs: everything stays rule-based)
        self.learning = learning
        # Step 5d approved self-healing rules: matching sentences are removed from web pages before any LLM reads them
        self.rulebook = rulebook
        self.planner = PlannerAgent(config)
        self.extract_agent = ExtractAgent(config)
        self.gap_agent = GapCheckAgent(config)
        self.writer_agent = WriterAgent(config)
        self.critic_agent = CriticAgent(config)
        # Shared by every job on this worker: free-tier tokens-per-minute limits are per account, not per job
        self.llm_slots = asyncio.Semaphore(max(1, config.llm_concurrency))

    # ── planning and search fan-out ──

    @traceable(run_type="chain", name="orchestrator:plan")
    async def plan_node(self, state: ResearchState) -> dict:
        await progress.emit("planning")
        plan, direct_reply, recency = await self.planner.run(state["topic"], state.get("session_history", []))
        if direct_reply:
            await progress.emit("direct_answer")
        else:
            await progress.emit("planned", sub_questions=[{"id": sq["id"], "question": sq["question"],
                                                           "kinds": sq["kinds"]} for sq in plan])
        return {"sub_questions": plan, "round_subq_ids": [sq["id"] for sq in plan], "direct_reply": direct_reply,
                "recency": recency}

    def _search_sends(self, state: ResearchState, web_limit: int) -> list[Send] | str:
        round_no = state["search_rounds"] + 1
        subqs = [sq for sq in state["sub_questions"] if sq["id"] in state["round_subq_ids"]]
        calls = plan_calls(subqs, web_limit)
        sends = [Send("search_branch", SearchTask(round=round_no, subq=sq, calls=calls[sq["id"]],
                                                  recency=state.get("recency", "")))
                 for sq in subqs if calls[sq["id"]]]
        return sends or "merge"

    def route_after_plan(self, state: ResearchState) -> list[Send] | str:
        if state.get("direct_reply"):
            return END
        return self._search_sends(state, self.config.search_max_queries)

    @traceable(run_type="chain", name="orchestrator:search_branch")
    async def search_branch(self, task: SearchTask) -> dict:
        subq = task["subq"]
        await progress.emit("searching", subq=subq["id"], question=subq["question"],
                            kinds=sorted({kind for _, kind, _ in task["calls"]}), queries=len(task["calls"]))
        # Injected searches (benchmark snapshots, attack runs) replay fixed results and take no date limit
        dated = {"recency": task["recency"]} if task.get("recency") and self.search_fn is None else {}
        outcomes = await asyncio.gather(
            *((self.search_fn or search_kind)(self.redis, self.config, q, kind, allow_paid=paid, **dated)
              for q, kind, paid in task["calls"]),
            return_exceptions=True,
        )
        ok = [o for o in outcomes if not isinstance(o, Exception)]
        errors = [str(o) for o in outcomes if isinstance(o, Exception)]
        for (q, kind, _), o in zip(task["calls"], outcomes):
            if isinstance(o, Exception):
                logger.warning(f"Sub-question {subq['id']}: '{q}' ({kind}) failed: {o}")
        results = [{**r, "subq": subq["id"]} for r in interleave(ok)]
        await progress.emit("searched", subq=subq["id"], results=len(results), failed=len(errors))
        return {"raw_results": [{"round": task["round"], "subq": subq["id"], "results": results,
                                 "error": errors[0] if errors and not ok else ""}]}

    @traceable(run_type="chain", name="orchestrator:merge")
    async def merge_node(self, state: ResearchState) -> dict:
        round_no = state["search_rounds"] + 1
        batch = sorted((r for r in state["raw_results"] if r["round"] == round_no), key=lambda r: r["subq"])
        existing = state["sources"]
        # Low-quality domains go first, so they never take a slot or a scrape
        unique = dedupe_new([r["results"] for r in batch], existing)
        candidates = drop_blocked(unique, self.config.blocked_domains)
        if len(candidates) < len(unique):
            await progress.emit("defense", what="blocked_domains", count=len(unique) - len(candidates))
        # A topic about today's news: pages clearly older than it asks for don't take a slot
        candidates, too_old = prefer_recent(candidates, state.get("recency", ""), date.today())
        # The first round leaves a few slots free, so a gap-check round can still add sources
        limit = self.config.search_max_sources
        if round_no == 1 and self.config.search_max_rounds > 1:
            limit = max(1, limit - _GAP_RESERVE)
        # Scrape only what can still make the cut, plus a few spares for pages that fail to scrape
        room = max(0, limit - len(existing))
        window = await enrich_sources(self.redis, self.config, candidates[:room + self.config.scrape_max_pages])
        if self.rulebook is not None and self.rulebook.patterns:
            from app.security.rules import sanitize_sources
            window = sanitize_sources(window, self.rulebook.patterns)
            cleaned = sum(1 for s in window if s.get("sanitized"))
            if cleaned:
                await progress.emit("defense", what="injection_rule", count=cleaned)
        # A GDELT entry is only a headline until its article is scraped - not worth citing on its own
        window = [s for s in window if s["provider"] != "gdelt" or s.get("scraped")]
        sources = merge_sources([window], existing, self.config, limit=limit)
        await progress.emit("sources", total=len(sources), new=len(sources) - len(existing),
                            read_in_full=sum(1 for s in sources if s.get("scraped")),
                            kinds=sorted({s.get("kind", "web") for s in sources}), too_old=too_old)
        update = {
            "sources": sources,
            "live_sources": bool(sources),
            "search_rounds": round_no,
            "scraped_count": sum(1 for s in sources if s.get("scraped")),
        }
        if not sources:
            errors = [r["error"] for r in batch if r["error"]]
            update["error"] = f"search: {errors[0] if errors else 'no usable results'}"
            logger.warning(f"Web search unavailable: {update['error']}")
        return update

    def route_after_merge(self, state: ResearchState) -> list[Send] | str:
        if not state["live_sources"]:
            return "summarize"
        current = set(state["round_subq_ids"])
        sends = []
        for sq in state["sub_questions"]:
            own = [s for s in state["sources"] if s.get("subq") == sq["id"]]
            if sq["id"] in current and own:
                sends.append(Send("extract_branch", ExtractTask(topic=state["topic"], subq=sq, sources=own)))
        return sends or "collect"

    # ── extraction ──

    @traceable(run_type="chain", name="orchestrator:extract_branch")
    async def extract_branch(self, task: ExtractTask) -> dict:
        async with self.llm_slots:
            await progress.emit("extracting", subq=task["subq"]["id"], sources=len(task["sources"]))
            facts = await self.extract_agent.run(task["topic"], task["sources"], task["subq"]["question"])
        await progress.emit("extracted", subq=task["subq"]["id"], facts=facts.count("\n- ") + facts.startswith("- "))
        return {"branch_facts": [{"subq": task["subq"]["id"], "question": task["subq"]["question"], "facts": facts}]}

    @traceable(run_type="chain", name="orchestrator:summarize")
    async def summarize_node(self, state: ResearchState) -> dict:
        return {"facts": await self.extract_agent.run(state["topic"], [])}

    async def collect_node(self, state: ResearchState) -> dict:
        return {"facts": combine_facts(state["branch_facts"])} if state["branch_facts"] else {}

    # ── gap check, writing, verification ──

    @traceable(run_type="chain", name="orchestrator:gap_check")
    async def gap_check_node(self, state: ResearchState) -> dict:
        if not state["live_sources"] or state["search_rounds"] >= self.config.search_max_rounds:
            return {"needs_more": False}
        if len(state["sources"]) >= self.config.search_max_sources:
            # Every source slot is taken: another round could only spend searches and an LLM call for nothing
            return {"needs_more": False}
        searched = [q for sq in state["sub_questions"] for q in sq["queries"]]
        queries, kinds = await self.gap_agent.run(state["topic"], state["facts"], searched)
        if not queries:
            return {"needs_more": False}
        logger.info(f"Gap check: searching again with {queries} ({kinds[0]})")
        await progress.emit("gap", queries=queries, kind=kinds[0])
        gap = SubQuestion(id=len(state["sub_questions"]) + 1, question=f"Additional findings: {queries[0]}",
                          queries=queries, kinds=kinds)
        return {"needs_more": True, "sub_questions": state["sub_questions"] + [gap], "round_subq_ids": [gap["id"]]}

    def route_after_gap_check(self, state: ResearchState) -> list[Send] | str:
        return self._search_sends(state, _GAP_QUERIES) if state["needs_more"] else "write"

    @traceable(run_type="chain", name="orchestrator:write")
    async def write_node(self, state: ResearchState) -> dict:
        # One variant per job (rewrites keep it), never in benchmark runs: those pin the strict writer
        variant = state.get("writer_variant") or (
            self.learning.choose_writer() if self.learning and not metering.active() else None) or ""
        await progress.emit("writing", draft=state.get("iterations", 0) + 1, revising=bool(state.get("critic_feedback")))
        report = await self.writer_agent.run(
            state["topic"],
            state["facts"],
            state["live_sources"],
            state.get("ltm_context", ""),
            state.get("critic_feedback", ""),
            variant=variant or None,
            recency=state.get("recency", ""),
        )
        if state["live_sources"]:
            report += sources_section(report, state["sources"])
        return {"report": report, "iterations": state.get("iterations", 0) + 1, "writer_variant": variant}

    @traceable(run_type="chain", name="orchestrator:verify")
    async def verify_node(self, state: ResearchState) -> dict:
        mode = self.config.critic_mode
        if mode in ("panel", "debate") and state["live_sources"]:
            from app.critics import run_panel
            await progress.emit("checking", mode=mode)
            trust, policy = self.learning.critic_learners() if self.learning else (None, None)
            result = await run_panel(self.config, state["report"], state["sources"], debate=(mode == "debate"),
                                     trust=trust, policy=policy)
            logger.info(f"Critic panel ({mode}): {sum(c['decision'] != 'keep' for c in result.claims)} of "
                        f"{len(result.claims)} claims need work, {sum(c['debated'] for c in result.claims)} debated")
            if self.learning:
                await self.learning.apply_trust_events(result.trust_events)
            await progress.emit("checked", mode=mode, passed=result.passed, claims=len(result.claims),
                                needs_work=sum(c["decision"] != "keep" for c in result.claims),
                                debated=sum(c["debated"] for c in result.claims))
            return {"verified": result.passed, "critic_feedback": result.feedback, "claim_verdicts": result.claims}
        await progress.emit("checking", mode="single")
        verified, feedback = await self.critic_agent.run(state["report"], state["sources"])
        await progress.emit("checked", mode="single", passed=verified,
                            needs_work=0 if verified else max(1, feedback.count("\n- ") + feedback.startswith("- ")))
        return {"verified": verified, "critic_feedback": feedback, "claim_verdicts": []}

    def route_after_verify(self, state: ResearchState) -> str:
        """Orchestrator decision: rewrite with critic feedback or finish."""
        if state.get("critic_feedback") == CRITIC_UNAVAILABLE:
            return END
        if not state["verified"] and state.get("iterations", 0) < self.config.agent_max_iterations:
            logger.info(f"Critic rejected report - rewriting (iteration {state['iterations']})")
            return "write"
        return END


def build_graph(config: Config, redis: aioredis.Redis, search_fn=None, learning=None, rulebook=None):
    orchestrator = OrchestratorAgent(config, redis, search_fn, learning, rulebook)
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", orchestrator.plan_node)
    workflow.add_node("search_branch", orchestrator.search_branch)
    workflow.add_node("merge", orchestrator.merge_node)
    workflow.add_node("extract_branch", orchestrator.extract_branch)
    workflow.add_node("summarize", orchestrator.summarize_node)
    workflow.add_node("collect", orchestrator.collect_node)
    workflow.add_node("gap_check", orchestrator.gap_check_node)
    workflow.add_node("write", orchestrator.write_node)
    workflow.add_node("verify", orchestrator.verify_node)

    workflow.add_edge(START, "plan")
    workflow.add_conditional_edges("plan", orchestrator.route_after_plan, ["search_branch", "merge", END])
    workflow.add_edge("search_branch", "merge")
    workflow.add_conditional_edges("merge", orchestrator.route_after_merge, ["extract_branch", "summarize", "collect"])
    workflow.add_edge("extract_branch", "collect")
    workflow.add_edge("summarize", "collect")
    workflow.add_edge("collect", "gap_check")
    workflow.add_conditional_edges("gap_check", orchestrator.route_after_gap_check, ["search_branch", "merge", "write"])
    workflow.add_edge("write", "verify")
    workflow.add_conditional_edges("verify", orchestrator.route_after_verify, {"write": "write", END: END})

    return workflow.compile()
