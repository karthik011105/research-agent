"""Fetches full article text for sources that only came with a headline or snippet.

The URLs come from search results, i.e. from the open internet, and the app runs inside a VPC next to
Redis, Postgres and the ECS metadata endpoint. So every hop (including redirects) must resolve only to
public IP addresses, on the standard web ports, and responses are capped in size and time.
"""
import asyncio
import hashlib
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlsplit
import httpx
import redis.asyncio as aioredis
import trafilatura
from langsmith import traceable
from app.config import Config

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "scrape:"
_FAILED = "\x00"                      # cached marker: page could not be fetched/extracted
_FAILED_TTL = 3600
_MAX_REDIRECTS = 3
_ALLOWED_PORTS = {80, 443}
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_USER_AGENT = "Mozilla/5.0 (compatible; ResearchAgentBot/1.0)"
# Kinds whose providers return only a headline/snippet and link to a normal article page.
# Papers (abstract already present, often PDF/JS pages) and HN discussions (comments already fetched) are skipped.
_SCRAPABLE_PROVIDERS = {"gdelt", "brave", "tavily"}


class BlockedURL(Exception):
    pass


async def _resolve(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def check_public_url(url: str) -> str:
    """Returns a public IP the URL's host resolves to. Raises BlockedURL unless the URL is http(s), has no
    credentials, uses a standard port, and every address the host resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise BlockedURL(f"unsupported URL: {url[:100]}")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise BlockedURL("invalid port")
    if port not in _ALLOWED_PORTS:
        raise BlockedURL(f"port {port} not allowed")
    try:
        addresses = await _resolve(parts.hostname)
    except OSError as e:
        raise BlockedURL(f"cannot resolve {parts.hostname}: {e}")
    if not addresses:
        raise BlockedURL(f"no address for {parts.hostname}")
    ips = [ipaddress.ip_address(addr.split("%")[0]) for addr in addresses]
    for ip in ips:
        if not ip.is_global or ip.is_multicast:
            raise BlockedURL(f"{parts.hostname} resolves to non-public address {ip}")
    # The VPC has no IPv6 route, so prefer IPv4 when the host offers both
    return str(next((ip for ip in ips if ip.version == 4), ips[0]))


def pinned_request(client: httpx.AsyncClient, url: str, ip: str) -> httpx.Request:
    """Connects to the IP that was checked (not a fresh DNS lookup, which could be rebound to an internal
    address), while Host header and TLS SNI/certificate checks still use the real hostname."""
    parts = urlsplit(url)
    host = f"[{ip}]" if ":" in ip else ip
    netloc = f"{host}:{parts.port}" if parts.port else host
    host_header = f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname
    return client.build_request("GET", parts._replace(netloc=netloc).geturl(),
                                headers={"Host": host_header}, extensions={"sni_hostname": parts.hostname})


async def fetch_html(url: str, config: Config) -> str:
    """GET with manual redirects so each hop is checked and pinned. Returns HTML text or raises."""
    async with httpx.AsyncClient(timeout=config.scrape_timeout, follow_redirects=False,
                                 headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"}) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            ip = await check_public_url(url)
            r = await client.send(pinned_request(client, url, ip), stream=True)
            try:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        raise httpx.HTTPError("redirect without location")
                    url = urljoin(url, location)
                    continue
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if not ctype.startswith(_HTML_TYPES):
                    raise ValueError(f"not HTML: {ctype or 'unknown type'}")
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > config.scrape_max_bytes:
                        raise ValueError("page too large")
                return body.decode(r.encoding or "utf-8", errors="replace")
            finally:
                await r.aclose()
    raise httpx.HTTPError("too many redirects")


def extract_text(page_html: str) -> str:
    text = trafilatura.extract(page_html, include_comments=False, include_tables=False, favor_precision=True)
    return " ".join((text or "").split())


async def scrape(redis: aioredis.Redis, config: Config, url: str) -> str:
    """Article text for a URL, '' if it can't be fetched. Successes and failures are both cached."""
    key = _CACHE_PREFIX + hashlib.sha256(url.encode()).hexdigest()
    cached = await redis.get(key)
    if cached is not None:
        return "" if cached == _FAILED else cached
    try:
        page = await fetch_html(url, config)
        # trafilatura is CPU-bound; keep the event loop free for other jobs
        text = (await asyncio.to_thread(extract_text, page))[:config.source_char_limit * 2]
    except (BlockedURL, httpx.HTTPError, ValueError, UnicodeError) as e:
        logger.info(f"Scrape skipped for {url[:120]}: {e}")
        text = ""
    if text:
        await redis.setex(key, config.search_cache_ttl, text)
    else:
        await redis.setex(key, _FAILED_TTL, _FAILED)
    return text


def needs_scrape(source: dict, config: Config) -> bool:
    return source.get("provider") in _SCRAPABLE_PROVIDERS and len(source.get("content", "")) < config.scrape_min_chars


@traceable(run_type="tool", name="tool:scrape")
async def enrich_sources(redis: aioredis.Redis, config: Config, sources: list[dict]) -> list[dict]:
    """Replaces thin content (headline/snippet) with the article text for up to scrape_max_pages sources.
    Order and everything else are kept; a failed scrape leaves the source as it was."""
    targets = [i for i, s in enumerate(sources) if needs_scrape(s, config)][:config.scrape_max_pages]
    if not targets:
        return sources
    texts = await asyncio.gather(*(scrape(redis, config, sources[i]["url"]) for i in targets))
    enriched = list(sources)
    for i, text in zip(targets, texts):
        if len(text) > len(sources[i]["content"]):
            enriched[i] = {**sources[i], "content": text, "scraped": True}
    return enriched
