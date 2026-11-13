"""Free, keyless data sources beyond general web search.

Each fetcher returns the shared source shape from search.make_source:
{title, url, content, published_date, provider, kind}.
Parsers are pure functions so they can be tested against saved API responses.
"""
import asyncio
import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
import httpx

ARXIV_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"
HN_DISCUSSION_URL = "https://news.ycombinator.com/item?id={id}"
# arXiv and Wikimedia ask API clients to identify themselves rather than use a library default
HEADERS = {"User-Agent": "research-agent/1.0 (LangGraph research assistant; httpx)"}

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_STOPWORDS = {
    "the", "and", "for", "with", "what", "are", "how", "new", "latest", "recent", "about", "from", "into",
    "does", "this", "that", "they", "their", "which", "who", "why", "when", "over", "under", "between",
}
_TAG = re.compile(r"<[^>]+>")

# The providers publish request-rate rules (arXiv: one call per 3 s, GDELT: one per 5 s, keyless Semantic
# Scholar: about one per second). One app task serves all jobs, so a per-process gap between calls is enough.
# search._free_search calls throttle() before each request, then checks the provider's cooldown.
_MIN_INTERVAL = {"arxiv": 3.0, "gdelt": 5.0, "semantic_scholar": 1.1}
_last_call: dict[str, float] = {}
_locks: dict[str, asyncio.Lock] = {}


async def throttle(provider: str) -> None:
    interval = _MIN_INTERVAL.get(provider)
    if not interval:
        return
    lock = _locks.setdefault(provider, asyncio.Lock())
    async with lock:
        wait = _last_call.get(provider, 0.0) + interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call[provider] = time.monotonic()


def _clean(text: str | None) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", text or "")).split())


def iso_date(value: str | None) -> str:
    """Providers send '2026-11-09', '2026-11-09T10:33:57Z' or 'Mon, 09 Nov 2026 10:33:57 GMT' -> '2026-11-09'."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        return value[:40]   # unknown format (e.g. Brave's "2 days ago"): keep as given


def make_source(title: str, url: str, content: str, published_date: str, provider: str, kind: str) -> dict:
    return {
        "title": _clean(title) or url.strip(),
        "url": url.strip(),
        "content": " ".join((content or "").split()),
        "published_date": iso_date(published_date),
        "provider": provider,
        "kind": kind,
    }


def keyword_terms(query: str, limit: int = 4) -> list[str]:
    """Search-engine style query → a few content words (paper search APIs work best on keywords)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2][:limit]


# ── arXiv ────────────────────────────────────────────────────────────────────

def arxiv_query(query: str) -> str:
    terms = keyword_terms(query)
    return " AND ".join(f"all:{t}" for t in terms) if terms else f"all:{query}"


def parse_arxiv(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("a:entry", _ATOM):
        url = (entry.findtext("a:id", "", _ATOM) or "").replace("http://", "https://", 1)
        if not url:
            continue
        authors = [a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)]
        byline = ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else "")
        abstract = _clean(entry.findtext("a:summary", "", _ATOM))
        content = f"Authors: {byline}. Abstract: {abstract}" if byline else abstract
        published = (entry.findtext("a:published", "", _ATOM) or "")[:10]
        results.append(make_source(entry.findtext("a:title", "", _ATOM), url, content, published, "arxiv", "paper"))
    return results


async def fetch_arxiv(query: str, max_results: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(ARXIV_URL, params={
            "search_query": arxiv_query(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        })
        r.raise_for_status()
        return parse_arxiv(r.text)


# ── Semantic Scholar ─────────────────────────────────────────────────────────

def parse_semantic_scholar(payload: dict) -> list[dict]:
    results = []
    for p in payload.get("data") or []:
        url = p.get("url") or (f"https://www.semanticscholar.org/paper/{p['paperId']}" if p.get("paperId") else "")
        if not url or not p.get("abstract"):
            continue  # title-only entries carry no facts to extract
        meta = [str(p["year"])] if p.get("year") else []
        if p.get("venue"):
            meta.append(p["venue"])
        if p.get("citationCount") is not None:
            meta.append(f"{p['citationCount']} citations")
        content = (f"{'; '.join(meta)}. " if meta else "") + f"Abstract: {_clean(p['abstract'])}"
        results.append(make_source(p.get("title", ""), url, content, p.get("publicationDate") or "", "semantic_scholar", "paper"))
    return results


async def fetch_semantic_scholar(query: str, max_results: int, api_key: str = "") -> list[dict]:
    headers = {"x-api-key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(SEMANTIC_SCHOLAR_URL, headers=headers, params={
            "query": " ".join(keyword_terms(query, 6)) or query,
            "limit": max_results,
            "fields": "title,abstract,url,year,venue,publicationDate,citationCount",
        })
        r.raise_for_status()
        return parse_semantic_scholar(r.json())


# ── OpenAlex (open scholarly index: 250M+ works, keyless, generous limits) ──

def openalex_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]} for licensing reasons; rebuild the text."""
    if not inverted:
        return ""
    positions = {pos: word for word, places in inverted.items() for pos in places}
    return " ".join(positions[i] for i in sorted(positions))


def parse_openalex(payload: dict) -> list[dict]:
    results, seen_titles = [], set()
    for w in payload.get("results") or []:
        title = _clean(w.get("title") or w.get("display_name"))
        abstract = openalex_abstract(w.get("abstract_inverted_index"))
        url = (w.get("primary_location") or {}).get("landing_page_url") or w.get("doi") or w.get("id") or ""
        key = title.lower()
        if not url or not abstract or key in seen_titles:
            continue  # OpenAlex often lists several versions (preprint, published) of one paper
        seen_titles.add(key)
        meta = [m for m in (str(w.get("publication_year") or ""),
                            ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "") if m]
        if w.get("cited_by_count") is not None:
            meta.append(f"{w['cited_by_count']} citations")
        content = (f"{'; '.join(meta)}. " if meta else "") + f"Abstract: {abstract}"
        results.append(make_source(title, url, content, w.get("publication_date") or "", "openalex", "paper"))
    return results


async def fetch_openalex(query: str, max_results: int, since: str = "") -> list[dict]:
    """since (YYYY-MM-DD) limits results to works published on or after that date."""
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(OPENALEX_URL, params={
            "search": " ".join(keyword_terms(query, 6)) or query,
            "filter": "has_abstract:true" + (f",from_publication_date:{since}" if since else ""),
            "per-page": max_results,
            "select": "id,doi,title,display_name,publication_date,publication_year,abstract_inverted_index,"
                      "cited_by_count,primary_location",
        })
        r.raise_for_status()
        return parse_openalex(r.json())


# ── GDELT (global news) ──────────────────────────────────────────────────────

def _gdelt_date(seen: str) -> str:
    try:
        return datetime.strptime(seen, "%Y%m%dT%H%M%SZ").date().isoformat()
    except (TypeError, ValueError):
        return ""


def parse_gdelt(payload: dict) -> list[dict]:
    """GDELT lists headlines only; the scraper fills in the article text later."""
    return [
        make_source(a.get("title", ""), a["url"], a.get("title", ""), _gdelt_date(a.get("seendate", "")), "gdelt", "news")
        for a in payload.get("articles") or []
        if a.get("url")
    ]


async def fetch_gdelt(query: str, max_results: int, timespan: str = "1m") -> list[dict]:
    terms = keyword_terms(query, 5)
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(GDELT_URL, params={
            "query": f"{' '.join(terms) or query} sourcelang:english",
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_results,
            "timespan": timespan,
            "sort": "hybridrel",
        })
        r.raise_for_status()
        try:
            return parse_gdelt(r.json())
        except ValueError:
            # GDELT answers bad queries (e.g. a word that is too short) with a plain-text message
            return []


# ── Hacker News (community discussion) ───────────────────────────────────────

def top_comments(item: dict, limit: int = 5, chars: int = 300) -> list[str]:
    comments = [c for c in item.get("children") or [] if c.get("type") == "comment" and c.get("text")]
    comments.sort(key=lambda c: len(c.get("children") or []), reverse=True)  # most-replied first
    return [_clean(c["text"])[:chars] for c in comments[:limit]]


def parse_hn(hit: dict, item: dict | None) -> dict:
    story_id = hit["objectID"]
    points, n_comments = hit.get("points") or 0, hit.get("num_comments") or 0
    parts = [f"Hacker News discussion ({points} points, {n_comments} comments)."]
    if hit.get("url"):
        parts.append(f"Linked article: {hit['url']}.")
    if hit.get("story_text"):
        parts.append(_clean(hit["story_text"])[:500])
    comments = top_comments(item) if item else []
    if comments:
        parts.append("Top comments: " + " | ".join(comments))
    return make_source(hit.get("title", ""), HN_DISCUSSION_URL.format(id=story_id), " ".join(parts),
                       (hit.get("created_at") or "")[:10], "hackernews", "community")


async def fetch_hackernews(query: str, max_results: int, since_ts: int = 0) -> list[dict]:
    """The discussion itself is the source: story + its most-replied comments. since_ts (Unix time) limits
    results to stories posted after it."""
    params = {"query": " ".join(keyword_terms(query, 5)) or query, "tags": "story", "hitsPerPage": max_results}
    if since_ts:
        params["numericFilters"] = f"created_at_i>{since_ts}"
    async with httpx.AsyncClient(timeout=30, headers=HEADERS) as client:
        r = await client.get(HN_SEARCH_URL, params=params)
        r.raise_for_status()
        hits = [h for h in r.json().get("hits", []) if h.get("objectID") and (h.get("num_comments") or 0) > 0]

        async def item(story_id: str) -> dict | None:
            try:
                resp = await client.get(HN_ITEM_URL.format(id=story_id))
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError):
                return None

        items = await asyncio.gather(*(item(h["objectID"]) for h in hits))
        return [parse_hn(h, i) for h, i in zip(hits, items)]
