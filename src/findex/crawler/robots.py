import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT


class RobotsCache:
    """Асинхронно завантажує, парсить і кешує robots.txt окремо для кожного хоста."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 5.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        """Перевіряє, чи дозволяє robots.txt обхід вказаного URL."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        if not scheme or not netloc:
            return False

        host_key = f"{scheme}://{netloc}"
        parser = self._parsers.get(host_key)
        if parser is None:
            lock = self._locks.setdefault(host_key, asyncio.Lock())
            async with lock:
                parser = self._parsers.get(host_key)
                if parser is None:
                    parser = await self._load_for_host(host_key)
                    self._parsers[host_key] = parser

        return parser.can_fetch(self._user_agent, url)

    async def _load_for_host(self, host_key: str) -> RobotFileParser:
        robots_url = f"{host_key}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            async with asyncio.timeout(self._timeout):
                resp = await self._client.get(
                    robots_url,
                    headers={"User-Agent": self._user_agent},
                    follow_redirects=True,
                )
            if 200 <= resp.status_code < 300:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403, 408, 429) or resp.status_code >= 500:
                rp.parse(["User-agent: *", "Disallow: /"])
            else:
                rp.parse([])
        except (TimeoutError, httpx.RequestError):
            rp.parse(["User-agent: *", "Disallow: /"])

        return rp
