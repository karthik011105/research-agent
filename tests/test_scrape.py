import httpx
import pytest
import respx
from app.tools import scrape
from app.tools.scrape import BlockedURL, check_public_url, enrich_sources, fetch_html, needs_scrape
from tests.conftest import make_source

PUBLIC_IP = "93.184.215.14"
ARTICLE = "<html><body><nav>Menu</nav><article><h1>Fines</h1>" + "<p>The regulator fined the AI firm 2 million euros for unsafe outputs.</p>" * 12 + "</article></body></html>"


@pytest.fixture(autouse=True)
def dns(monkeypatch):
    """Fake resolver: hostnames map to fixed addresses, so no test touches the network."""
    table = {
        "news.example": [PUBLIC_IP],
        "evil.example": ["10.0.0.5"],
        "mixed.example": [PUBLIC_IP, "127.0.0.1"],
        "metadata.example": ["169.254.170.2"],
    }

    async def resolve(host):
        if host in table:
            return table[host]
        if host.replace(".", "").isdigit():
            return [host]
        raise OSError("no such host")

    monkeypatch.setattr(scrape, "_resolve", resolve)


@pytest.mark.parametrize("url", [
    "http://evil.example/x",            # private range
    "http://mixed.example/",            # one public + one loopback address
    "http://metadata.example/v2/credentials",  # ECS task metadata
    "http://127.0.0.1/",
    "http://news.example:6379/",        # Redis port
    "ftp://news.example/file",
    "http://user:pw@news.example/",
    "http://unknown.example/",
])
async def test_check_public_url_blocks_internal_targets(url):
    with pytest.raises(BlockedURL):
        await check_public_url(url)


async def test_check_public_url_allows_public_host():
    assert await check_public_url("https://news.example/a?b=1") == PUBLIC_IP


async def test_check_public_url_prefers_ipv4(monkeypatch):
    async def dual_stack(host):
        return ["2606:4700::6810:84e5", PUBLIC_IP]

    monkeypatch.setattr(scrape, "_resolve", dual_stack)
    assert await check_public_url("https://dual.example/") == PUBLIC_IP


@respx.mock
async def test_fetch_connects_to_checked_ip_with_real_host_header(config):
    route = respx.get(f"https://{PUBLIC_IP}/a").mock(
        return_value=httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html; charset=utf-8"}))
    assert "regulator fined" in await fetch_html("https://news.example/a", config)
    request = route.calls[0].request
    assert request.headers["host"] == "news.example"
    assert request.extensions["sni_hostname"] == "news.example"


@respx.mock
async def test_redirect_to_private_address_is_blocked(config):
    respx.get(f"http://{PUBLIC_IP}/a").mock(return_value=httpx.Response(302, headers={"location": "http://evil.example/admin"}))
    with pytest.raises(BlockedURL):
        await fetch_html("http://news.example/a", config)


@respx.mock
async def test_redirect_chain_is_capped(config):
    respx.get(f"http://{PUBLIC_IP}/loop").mock(return_value=httpx.Response(302, headers={"location": "/loop"}))
    with pytest.raises(httpx.HTTPError, match="too many redirects"):
        await fetch_html("http://news.example/loop", config)


@respx.mock
async def test_non_html_and_oversized_pages_are_rejected(config):
    respx.get(f"http://{PUBLIC_IP}/file.pdf").mock(return_value=httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}))
    respx.get(f"http://{PUBLIC_IP}/big").mock(return_value=httpx.Response(200, content=b"<p>" + b"x" * 300_000, headers={"content-type": "text/html"}))
    with pytest.raises(ValueError, match="not HTML"):
        await fetch_html("http://news.example/file.pdf", config)
    with pytest.raises(ValueError, match="too large"):
        await fetch_html("http://news.example/big", config)


def test_needs_scrape_only_for_thin_article_sources(config):
    headline = make_source(1, content="Regulator fines AI firm", provider="gdelt", kind="news")
    paper = make_source(2, content="short abstract", provider="arxiv", kind="paper")
    full = make_source(3)
    assert needs_scrape(headline, config) and not needs_scrape(paper, config) and not needs_scrape(full, config)


@respx.mock
async def test_enrich_replaces_thin_content_and_caches_both_outcomes(config, redis):
    ok = respx.get(f"https://{PUBLIC_IP}/a").mock(
        return_value=httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"}))
    sources = [
        make_source(1, url="https://news.example/a", content="Regulator fines AI firm", provider="gdelt", kind="news"),
        make_source(2, url="http://evil.example/b", content="Internal page", provider="gdelt", kind="news"),
        make_source(3),
    ]
    enriched = await enrich_sources(redis, config, sources)
    assert enriched[0]["scraped"] is True and "2 million euros" in enriched[0]["content"]
    assert enriched[1] == sources[1]           # blocked: left unchanged
    assert enriched[2] == sources[2]           # already full text: not fetched
    await enrich_sources(redis, config, sources)
    assert ok.call_count == 1                  # second run served from cache


async def test_enrich_respects_page_limit(config, redis, monkeypatch):
    config.scrape_max_pages = 2
    fetched = []

    async def fake_scrape(redis_, config_, url):
        fetched.append(url)
        return "Full article text " * 50

    monkeypatch.setattr(scrape, "scrape", fake_scrape)
    thin = [make_source(i, content="headline", provider="gdelt", kind="news") for i in range(5)]
    enriched = await enrich_sources(redis, config, thin)
    assert len(fetched) == 2 and sum(1 for s in enriched if s.get("scraped")) == 2
