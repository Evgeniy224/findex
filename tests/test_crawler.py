import asyncio
import json
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler.crawl import CrawlStats, canonicalize_url, crawl
from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.index import build_index


def test_canonicalize_url() -> None:
    assert (
        canonicalize_url("HTTPS://Example.COM:443/docs/intro/?b=2&a=1#section")
        == "https://example.com/docs/intro?a=1&b=2"
    )
    assert canonicalize_url("/guide/setup/#top", base_url="https://example.com/docs/intro") == (
        "https://example.com/guide/setup"
    )
    assert canonicalize_url("#only-fragment", base_url="https://example.com") is None
    assert canonicalize_url("mailto:test@example.com") is None


def test_fetch_retries_on_429_and_honors_retry_after() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Too Many Requests")
        if attempts == 2:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, text="<html><title>OK</title><body>Hello</body></html>")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/page",
                timeout=2.0,
                max_retries=3,
                backoff_base=0.01,
                jitter_max=0.0,
            )
        assert res.status == 200
        assert res.attempts == 3
        assert res.error is None
        assert "Hello" in res.text

    asyncio.run(_run())
    assert attempts == 3


def test_fetch_does_not_retry_404() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="Not Found")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/missing",
                timeout=2.0,
                max_retries=3,
                backoff_base=0.01,
                jitter_max=0.0,
            )
        assert res.status == 404
        assert res.attempts == 1
        assert res.error == "HTTP 404"

    asyncio.run(_run())
    assert attempts == 1


def test_fetch_honors_retry_after_without_truncating_it(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    attempts = 0

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "12"})
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        monkeypatch.setattr(asyncio, "sleep", record_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.com/",
                max_retries=2,
                backoff_cap=1,
                jitter_max=0,
            )
        assert result.status == 200

    asyncio.run(_run())
    assert sleeps == [12.0]


def test_fetch_checks_redirect_policy_before_following() -> None:
    requests: list[str] = []
    checked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://outside.example/page"})

    async def deny_redirect(url: str) -> bool:
        checked.append(url)
        return False

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(
                client,
                "https://example.com/start",
                redirect_allowed=deny_redirect,
            )
        assert result.status == 302
        assert result.error == "Redirect blocked by crawl policy: https://outside.example/page"

    asyncio.run(_run())
    assert requests == ["https://example.com/start"]
    assert checked == ["https://outside.example/page"]


def test_fetch_retries_on_timeout_up_to_max_retries() -> None:
    attempts = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.2)
        return httpx.Response(200, text="Late")

    async def _run() -> None:
        transport = httpx.MockTransport(slow_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/slow",
                timeout=0.03,
                max_retries=3,
                backoff_base=0.01,
                jitter_max=0.0,
            )
        assert res.status == 0
        assert res.attempts == 3
        assert res.error is not None and "Timeout" in res.error

    asyncio.run(_run())
    assert attempts == 3


def test_robots_cache_per_host() -> None:
    robots_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_requests
        if request.url.path == "/robots.txt":
            robots_requests += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="OK")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://example.com/public/page") is True
            assert await cache.allowed("https://example.com/private/secret") is False
            assert await cache.allowed("https://example.com/public/other") is True

    asyncio.run(_run())
    assert robots_requests == 1


@pytest.mark.parametrize("status", [401, 403, 408, 429, 503])
def test_robots_cache_fails_closed_on_access_or_server_errors(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://example.com/public") is False

    asyncio.run(_run())


def test_robots_cache_allows_when_file_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def _run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://example.com/public") is True

    asyncio.run(_run())


def test_crawl_generator_deduplicates_obeys_robots_and_stops_at_max_pages(tmp_path: Path) -> None:
    pages_map: dict[str, str] = {
        "/robots.txt": "User-agent: *\nDisallow: /blocked\n",
        "/": (
            "<html><head><title>Home</title></head><body>Main index "
            '<a href="/a">A</a> '
            '<a href="/a#dup">A dup</a> '
            '<a href="/b">B</a> '
            '<a href="/blocked">Blocked</a> '
            '<a href="https://other.org/ext">External</a>'
            "</body></html>"
        ),
        "/a": (
            "<html><head><title>Page A</title></head><body>Alpha content "
            '<a href="/">Back</a> <a href="/c">C</a></body></html>'
        ),
        "/b": (
            "<html><head><title>Page B</title></head><body>Beta content "
            '<a href="/d">D</a></body></html>'
        ),
        "/c": "<html><head><title>Page C</title></head><body>Gamma content</body></html>",
        "/d": "<html><head><title>Page D</title></head><body>Delta content</body></html>",
        "/blocked": "<html><head><title>Secret</title></head><body>Should not crawl</body></html>",
    }
    fetched_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path != "/robots.txt":
            fetched_urls.append(str(request.url))
        if path in pages_map:
            return httpx.Response(200, text=pages_map[path])
        return httpx.Response(404, text="Not Found")

    stats = CrawlStats()

    async def _run() -> list[dict[str, str]]:
        transport = httpx.MockTransport(handler)
        collected: list[dict[str, str]] = []
        async for page in crawl(
            ["https://docs.example.com/"],
            max_pages=3,
            concurrency=2,
            per_host=2,
            min_delay=0.0,
            transport=transport,
            stats=stats,
        ):
            collected.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
            )
        return collected

    pages = asyncio.run(_run())
    assert len(pages) == 3
    assert stats.queue_depth == 0
    assert stats.in_flight == 0
    assert all("/blocked" not in p["url"] for p in pages)
    assert all("other.org" not in p["url"] for p in pages)
    assert len(set(fetched_urls)) == len(fetched_urls)
    assert len(fetched_urls) == 3

    jsonl_path = tmp_path / "crawled.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for item in pages:
            fh.write(json.dumps(item) + "\n")

    docs = list(iter_documents(jsonl_path))
    assert len(docs) == 3
    idx = build_index(jsonl_path)
    assert idx.num_docs == 3


def test_crawl_does_not_follow_cross_host_redirect() -> None:
    requested: list[str] = []
    records: list[FetchResult] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        requested.append(str(request.url))
        if request.url.host == "docs.example.com":
            return httpx.Response(
                302,
                headers={"Location": "https://external.example/page"},
            )
        return httpx.Response(200, text="<title>External</title>")

    async def _run() -> list[str]:
        found_urls: list[str] = []
        async for page in crawl(
            ["https://docs.example.com/"],
            max_pages=1,
            concurrency=1,
            min_delay=0,
            transport=httpx.MockTransport(handler),
            log_records=records,
        ):
            found_urls.append(page.url)
        return found_urls

    found_urls = asyncio.run(_run())
    assert found_urls == []
    assert requested == ["https://docs.example.com/"]
    assert len(records) == 1
    assert records[0].error is not None
    assert "Redirect blocked by crawl policy" in records[0].error


def test_crawl_with_empty_domain_allowlist_fetches_nothing() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, text="<title>should not fetch</title>")

    async def _run() -> list[str]:
        found_urls: list[str] = []
        async for page in crawl(
            ["https://docs.example.com/"],
            max_pages=1,
            allowed_domains=[],
            transport=httpx.MockTransport(handler),
        ):
            found_urls.append(page.url)
        return found_urls

    assert asyncio.run(_run()) == []
    assert requested == []


def test_crawl_logs_404_once_and_continues() -> None:
    requested: list[str] = []
    records: list[FetchResult] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        requested.append(str(request.url))
        return httpx.Response(404, text="Not Found")

    async def _run() -> list[str]:
        found_urls: list[str] = []
        async for page in crawl(
            ["https://docs.example.com/missing"],
            max_pages=1,
            concurrency=1,
            min_delay=0,
            transport=httpx.MockTransport(handler),
            log_records=records,
        ):
            found_urls.append(page.url)
        return found_urls

    assert asyncio.run(_run()) == []
    assert requested == ["https://docs.example.com/missing"]
    assert len(records) == 1
    assert records[0].status == 404
    assert records[0].attempts == 1
    assert records[0].error == "HTTP 404"