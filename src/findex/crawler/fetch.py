import asyncio
import email.utils
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

import httpx

DEFAULT_USER_AGENT = (
    "findex-crawler/0.6 (+https://github.com/Evgeniy224/findex; educational search engine lab)"
)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    num_bytes: int
    elapsed: float
    attempts: int
    error: str | None = None


def parse_retry_after(header_value: str | None) -> float | None:
    """Розбирає заголовок Retry-After (секунди або HTTP-дата) у затримку в секундах."""
    if not header_value:
        return None
    raw = header_value.strip()
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        parsed_dt = email.utils.parsedate_to_datetime(raw)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=UTC)
        delta = (parsed_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 8.0,
    jitter_max: float = 0.5,
    user_agent: str = DEFAULT_USER_AGENT,
    redirect_allowed: Callable[[str], Awaitable[bool]] | None = None,
    max_redirects: int = 20,
) -> FetchResult:
    """
    Виконує асинхронний HTTP GET із asyncio.timeout та повторами (до max_retries)
    лише для 429, 5xx та мережевих помилок/таймаутів з exponential backoff + jitter.
    """
    t0 = time.perf_counter()
    headers = {"User-Agent": user_agent}
    last_status = 0
    last_error: str | None = None
    last_final_url = url

    for attempt in range(1, max_retries + 1):
        retry_delay: float | None = None
        request_url = url
        visited_urls = {request_url}

        try:
            for _ in range(max_redirects + 1):
                async with asyncio.timeout(timeout):
                    response = await client.get(
                        request_url,
                        headers=headers,
                        follow_redirects=False,
                    )

                last_status = response.status_code
                last_final_url = str(response.url)
                raw_bytes = response.content

                if response.status_code not in REDIRECT_STATUS_CODES:
                    break

                location = response.headers.get("Location")
                if not location:
                    return FetchResult(
                        url=url,
                        final_url=last_final_url,
                        status=response.status_code,
                        text="",
                        num_bytes=len(raw_bytes),
                        elapsed=time.perf_counter() - t0,
                        attempts=attempt,
                        error="Redirect without Location header",
                    )

                next_url = urljoin(last_final_url, location)
                if redirect_allowed is not None and not await redirect_allowed(next_url):
                    return FetchResult(
                        url=url,
                        final_url=last_final_url,
                        status=response.status_code,
                        text="",
                        num_bytes=len(raw_bytes),
                        elapsed=time.perf_counter() - t0,
                        attempts=attempt,
                        error=f"Redirect blocked by crawl policy: {next_url}",
                    )
                if next_url in visited_urls:
                    return FetchResult(
                        url=url,
                        final_url=last_final_url,
                        status=response.status_code,
                        text="",
                        num_bytes=len(raw_bytes),
                        elapsed=time.perf_counter() - t0,
                        attempts=attempt,
                        error="Redirect loop",
                    )
                visited_urls.add(next_url)
                request_url = next_url
            else:
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=last_status,
                    text="",
                    num_bytes=0,
                    elapsed=time.perf_counter() - t0,
                    attempts=attempt,
                    error=f"Too many redirects (maximum {max_redirects})",
                )

            if 200 <= response.status_code < 300:
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=response.status_code,
                    text=response.text,
                    num_bytes=len(raw_bytes),
                    elapsed=time.perf_counter() - t0,
                    attempts=attempt,
                    error=None,
                )

            # 4xx (окрім 429) не повторюємо
            if 400 <= response.status_code < 500 and response.status_code != 429:
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=response.status_code,
                    text="",
                    num_bytes=len(raw_bytes),
                    elapsed=time.perf_counter() - t0,
                    attempts=attempt,
                    error=f"HTTP {response.status_code}",
                )

            if response.status_code in RETRYABLE_STATUS_CODES or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if response.status_code == 429:
                    retry_after = parse_retry_after(response.headers.get("Retry-After"))
                    if retry_after is not None:
                        retry_delay = retry_after
            else:
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=response.status_code,
                    text="",
                    num_bytes=len(raw_bytes),
                    elapsed=time.perf_counter() - t0,
                    attempts=attempt,
                    error=f"HTTP {response.status_code}",
                )

        except TimeoutError:
            last_status = 0
            last_error = f"Timeout after {timeout}s"
        except httpx.RequestError as exc:
            last_status = 0
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            if retry_delay is None:
                exp = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, jitter_max) if jitter_max > 0 else 0.0
                retry_delay = exp + jitter
            await asyncio.sleep(float(retry_delay))

    return FetchResult(
        url=url,
        final_url=last_final_url,
        status=last_status,
        text="",
        num_bytes=0,
        elapsed=time.perf_counter() - t0,
        attempts=max_retries,
        error=last_error or "Unknown fetch error",
    )
