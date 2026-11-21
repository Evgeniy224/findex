import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_sec(self) -> float:
        return self.fetched / self.elapsed


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """
    Канонізує URL:
    - розкриває відносні посилання відносно base_url;
    - залишає лише http/https;
    - приводить scheme і host до нижнього регістру;
    - видаляє фрагмент (#...);
    - нормалізує шлях і кінцевий слеш;
    - сортує query-параметри.
    """
    raw = url.strip()
    if not raw or raw.startswith(("#", "mailto:", "javascript:", "tel:")):
        return None

    resolved = urljoin(base_url, raw) if base_url else raw
    parts = urlsplit(resolved)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return None

    hostname = parts.hostname.lower()
    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443) or port is None:
        netloc = hostname
    else:
        netloc = f"{hostname}:{port}"

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)

    return urlunsplit((scheme, netloc, path, query, ""))


class _HTMLExtractor(HTMLParser):
    """Витягує title, видимий текст (ігноруючи script/style) та href посилання."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self._title_chunks: list[str] = []
        self._text_chunks: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript"):
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._in_title = True
        if tag_lower == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    self.links.append(attr_val)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag_lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self._title_chunks.append(cleaned)
        else:
            self._text_chunks.append(cleaned)

    @property
    def title(self) -> str:
        return " ".join(self._title_chunks).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_chunks).strip()


def extract_page_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Повертає (title, clean_text, canonical_links) зі сторінки HTML."""
    parser = _HTMLExtractor()
    parser.feed(html)
    parser.close()

    canonical_links: list[str] = []
    for raw_href in parser.links:
        canon = canonicalize_url(raw_href, base_url=base_url)
        if canon is not None:
            canonical_links.append(canon)
    return parser.title, parser.text, canonical_links


def _is_domain_allowed(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    host = (urlsplit(url).hostname or "").lower()
    return any(host == dom or host.endswith(f".{dom}") for dom in allowed_domains)


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.2,
    timeout: float = 10.0,
    allowed_domains: Iterable[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    log_records: list[FetchResult] | None = None,
) -> AsyncIterator[Page]:
    """
    Асинхронний генератор обходу веб-сторінок.
    Використовує asyncio.Queue, asyncio.TaskGroup, глобальний та per-host семафори,
    per-host мінімальну паузу, RobotsCache і коректне скасування на max_pages.
    """
    if max_pages <= 0:
        return

    seed_list = [c for s in seeds if (c := canonicalize_url(s)) is not None]
    if not seed_list:
        return

    if allowed_domains is not None:
        domains = {d.strip().lower().lstrip(".") for d in allowed_domains if d.strip()}
    else:
        domains = {(urlsplit(s).hostname or "").lower() for s in seed_list if urlsplit(s).hostname}

    crawl_stats = stats if stats is not None else CrawlStats()
    crawl_stats.started_at = time.perf_counter()

    frontier: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()
    for s in seed_list:
        if _is_domain_allowed(s, domains) and s not in seen:
            seen.add(s)
            frontier.put_nowait(s)
    crawl_stats.queue_depth = frontier.qsize()

    if frontier.empty():
        return

    out_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    last_host_req: dict[str, float] = {}
    stop_event = asyncio.Event()
    pages_emitted = 0

    client_kwargs: dict[str, object] = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport

    async with httpx.AsyncClient(**client_kwargs) as client:  # type: ignore[arg-type]
        robots = RobotsCache(client, user_agent=user_agent, timeout=min(5.0, timeout))

        async def _respect_host_politeness(host: str) -> None:
            if min_delay <= 0:
                return
            lock = host_locks.setdefault(host, asyncio.Lock())
            async with lock:
                now = time.perf_counter()
                prev = last_host_req.get(host)
                if prev is not None:
                    wait_for = min_delay - (now - prev)
                    if wait_for > 0:
                        await asyncio.sleep(wait_for)
                last_host_req[host] = time.perf_counter()

        async def _worker() -> None:
            nonlocal pages_emitted
            while not stop_event.is_set():
                try:
                    url = await asyncio.wait_for(frontier.get(), timeout=0.5)
                except TimeoutError:
                    if stop_event.is_set() or (frontier.empty() and crawl_stats.in_flight == 0):
                        return
                    continue

                crawl_stats.in_flight += 1
                crawl_stats.queue_depth = frontier.qsize()
                try:
                    if stop_event.is_set():
                        return

                    if not await robots.allowed(url):
                        res = FetchResult(
                            url=url,
                            final_url=url,
                            status=0,
                            text="",
                            num_bytes=0,
                            elapsed=0.0,
                            attempts=0,
                            error="Disallowed by robots.txt",
                        )
                        if log_records is not None:
                            log_records.append(res)
                        continue

                    host = (urlsplit(url).hostname or "").lower()
                    host_sem = host_sems.setdefault(host, asyncio.Semaphore(max(1, per_host)))

                    async with global_sem, host_sem:
                        if stop_event.is_set():
                            return
                        await _respect_host_politeness(host)
                        res = await fetch(
                            client,
                            url,
                            timeout=timeout,
                            user_agent=user_agent,
                        )

                    if log_records is not None:
                        log_records.append(res)

                    if res.error is not None or not (200 <= res.status < 300):
                        crawl_stats.errors += 1
                        continue

                    title, clean_text, links = extract_page_and_links(
                        res.text, base_url=res.final_url
                    )

                    if stop_event.is_set() or pages_emitted >= max_pages:
                        stop_event.set()
                        return

                    pages_emitted += 1
                    crawl_stats.fetched = pages_emitted

                    page = Page(
                        url=res.final_url,
                        title=title,
                        text=clean_text,
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                    await out_queue.put(page)

                    if pages_emitted >= max_pages:
                        stop_event.set()
                        return

                    for link in links:
                        if (
                            not stop_event.is_set()
                            and link not in seen
                            and _is_domain_allowed(link, domains)
                        ):
                            seen.add(link)
                            frontier.put_nowait(link)
                    crawl_stats.queue_depth = frontier.qsize()

                finally:
                    crawl_stats.in_flight -= 1
                    frontier.task_done()
                    crawl_stats.queue_depth = frontier.qsize()

        async def _supervisor() -> None:
            try:
                async with asyncio.TaskGroup() as tg:
                    workers = [tg.create_task(_worker()) for _ in range(max(1, concurrency))]
                    while not stop_event.is_set():
                        if frontier.empty() and crawl_stats.in_flight == 0:
                            stop_event.set()
                            break
                        await asyncio.sleep(0.05)

                    for w in workers:
                        if not w.done():
                            w.cancel()
            except asyncio.CancelledError:
                pass
            finally:
                await out_queue.put(None)

        sup_task = asyncio.create_task(_supervisor())
        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                yield item
        finally:
            stop_event.set()
            if not sup_task.done():
                sup_task.cancel()
            await asyncio.gather(sup_task, return_exceptions=True)