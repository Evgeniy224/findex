import argparse
import asyncio
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import TypedDict
from urllib.parse import urljoin

import httpx


class BenchmarkResult(TypedDict):
    name: str
    requests: int
    concurrency: int
    rps: float
    p50: float
    p95: float
    p99: float


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


async def _run_load(
    name: str,
    base_url: str,
    requests: int,
    concurrency: int,
    timeout: float,
) -> BenchmarkResult:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    latencies: list[float] = []
    errors: list[str] = []
    next_request = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=limits,
    ) as client:
        for _ in range(min(concurrency, requests)):
            response = await client.get("/health")
            response.raise_for_status()

        async def worker() -> None:
            nonlocal next_request
            while True:
                async with lock:
                    if next_request >= requests:
                        return
                    request_id = next_request
                    next_request += 1
                started = time.perf_counter()
                try:
                    response = await client.get("/search", params={"q": "python", "k": 10})
                    if response.status_code != 200:
                        errors.append(f"HTTP {response.status_code}")
                        continue
                    latencies.append((time.perf_counter() - started) * 1000)
                except httpx.HTTPError as exc:
                    errors.append(f"request {request_id}: {exc}")

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(min(concurrency, requests))))
        elapsed = time.perf_counter() - started

    if errors:
        raise RuntimeError(f"{name}: {len(errors)} requests failed: {errors[:3]}")
    if not latencies:
        raise RuntimeError(f"{name}: no successful requests")
    return {
        "name": name,
        "requests": len(latencies),
        "concurrency": concurrency,
        "rps": len(latencies) / elapsed,
        "p50": median(latencies),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until_ready(process: subprocess.Popen[bytes], url: str) -> None:
    deadline = time.monotonic() + 30
    async with httpx.AsyncClient(timeout=1) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Uvicorn exited with status {process.returncode}")
            try:
                response = await client.get(urljoin(url, "/health"))
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f"Server did not become ready at {url}")


async def _run_local(
    name: str,
    mode: str,
    workers: int,
    index_path: Path,
    requests: int,
    concurrency: int,
    timeout: float,
) -> BenchmarkResult:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["INDEX_PATH"] = str(index_path.resolve())
    env["BENCHMARK_MODE"] = mode
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "findex.web.benchmark_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await _wait_until_ready(process, url)
        return await _run_load(name, url, requests, concurrency, timeout)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    print("| Target | Requests | Concurrency | RPS | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['name']} | {result['requests']} | {result['concurrency']} "
            f"| {result['rps']:.1f} | {result['p50']:.2f} | "
            f"{result['p95']:.2f} | {result['p99']:.2f} |"
        )


async def _main(args: argparse.Namespace) -> None:
    index_path = args.index.resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")
    results = [
        await _run_local(
            "local, 1 worker, async def",
            "async",
            1,
            index_path,
            args.requests,
            args.concurrency,
            args.timeout,
        ),
        await _run_local(
            "local, 1 worker, def",
            "def",
            1,
            index_path,
            args.requests,
            args.concurrency,
            args.timeout,
        ),
        await _run_local(
            "local, 4 workers, def",
            "def",
            4,
            index_path,
            args.requests,
            args.concurrency,
            args.timeout,
        ),
    ]
    if args.url:
        results.append(
            await _run_load(
                f"deployed: {args.url}",
                args.url.rstrip("/"),
                args.deployed_requests,
                args.deployed_concurrency,
                args.timeout,
            )
        )
    _print_table(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark findex HTTP search latency.")
    parser.add_argument("--index", type=Path, default=Path("index.json"))
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--deployed-requests", type=int, default=300)
    parser.add_argument("--deployed-concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--url", help="Optional deployed base URL, e.g. https://findex.onrender.com")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
