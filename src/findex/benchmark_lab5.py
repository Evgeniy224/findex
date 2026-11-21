# pyright: reportUnknownMemberType=false
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from findex.index import ExecutorType, Index, build_partial, merge_indexes


def _is_gil_enabled() -> bool:
    """Безпечно перевіряє, чи увімкнено GIL (CPython 3.13+)."""
    check = getattr(sys, "_is_gil_enabled", None)
    return bool(check()) if callable(check) else True


def split_chunks(paths: list[Path], num_chunks: int) -> list[tuple[int, list[Path]]]:
    """Ділить список шляхів на num_chunks шматків із неперетинними start_doc_id."""
    if not paths:
        return []
    k = max(1, min(num_chunks, len(paths)))
    chunk_size = (len(paths) + k - 1) // k
    return [
        (start, paths[start : start + chunk_size]) for start in range(0, len(paths), chunk_size)
    ]


def get_cpu_info() -> str:
    """Повертає модель процесора на macOS/Linux та кількість логічних ядер."""
    model = platform.processor() or "Unknown CPU"
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            if out:
                model = out
        except subprocess.SubprocessError:
            pass
    return f"{model} ({os.cpu_count()} логічних ядер)"


def _rss_mb() -> float:
    """Повертає піковий RSS поточного процесу в МБ (на macOS ru_maxrss у байтах)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def measure_once(
    paths: list[Path],
    executor: ExecutorType,
    workers: int,
) -> tuple[float, float, float, float]:
    """
    Виконує один прогін індексації.
    Повертає: (wall_time_s, cpu_time_s, parent_rss_mb, merge_time_s).
    """
    chunks = split_chunks(paths, workers if executor != "serial" else 1)

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    if executor == "serial":
        partials: list[Index] = [build_partial(cp, sid) for sid, cp in chunks]
    else:
        pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
        ordered: list[Index | None] = [None] * len(chunks)
        with pool_cls(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(build_partial, cp, sid): idx for idx, (sid, cp) in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                ordered[future_to_idx[fut]] = fut.result()
        partials = [p for p in ordered if p is not None]

    t0_merge = time.perf_counter()
    _ = merge_indexes(partials)
    merge_time = time.perf_counter() - t0_merge

    wall_time = time.perf_counter() - t0_wall
    cpu_time = time.process_time() - t0_cpu
    rss_mb = _rss_mb()

    return wall_time, cpu_time, rss_mb, merge_time


def run_race_demo() -> None:
    """Демонстрація гонитви даних при спільному несинхронізованому лічильнику."""
    shared_counter = 0
    iterations = 200_000
    num_threads = 4

    def worker() -> None:
        nonlocal shared_counter
        for _ in range(iterations):
            tmp = shared_counter
            shared_counter = tmp + 1

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    expected = iterations * num_threads
    lost = expected - shared_counter
    print(f"\n--- Демонстрація гонитви даних (sys._is_gil_enabled() = {_is_gil_enabled()}) ---")
    print(f"Очікуване значення лічильника: {expected}")
    print(f"Фактичне значення лічильника:  {shared_counter} (втрачено: {lost})")


def main() -> None:
    data_dir = Path("data")
    paths = sorted(data_dir.rglob("*.txt"))
    cpu_cores = os.cpu_count() or 4
    worker_counts = sorted({1, 2, 4, 8, cpu_cores})
    repeats = 3

    print(f"CPU: {get_cpu_info()}")
    print(f"Python: {sys.version.split()[0]} | sys._is_gil_enabled() = {_is_gil_enabled()}")
    print(f"Документів у корпусі: {len(paths)} | Повторів на клітинку: {repeats}\n")

    # Прогрів кешу ОС
    _ = measure_once(paths, "serial", 1)

    # Базовий serial замір
    serial_runs = [measure_once(paths, "serial", 1) for _ in range(repeats)]
    serial_wall = statistics.median(r[0] for r in serial_runs)
    serial_cpu = statistics.median(r[1] for r in serial_runs)
    serial_rss = statistics.median(r[2] for r in serial_runs)
    serial_merge = statistics.median(r[3] for r in serial_runs)

    print(
        f"| {'Executor':<10} | {'Workers':>7} | {'Wall (s)':>9} | {'CPU (s)':>8} | "
        f"{'RSS (MB)':>9} | {'Merge (ms)':>10} | {'Speedup':>8} |"
    )
    print(f"|{'-' * 12}|{'-' * 9}|{'-' * 11}|{'-' * 10}|{'-' * 11}|{'-' * 12}|{'-' * 10}|")
    print(
        f"| {'serial':<10} | {1:>7} | {serial_wall:>9.4f} | {serial_cpu:>8.4f} | "
        f"{serial_rss:>9.1f} | {serial_merge * 1000:>10.2f} | {1.00:>7.2f}x |"
    )

    results: dict[str, dict[int, float]] = {"threads": {}, "processes": {}}

    for exec_type in ("threads", "processes"):
        for w in worker_counts:
            runs = [measure_once(paths, exec_type, w) for _ in range(repeats)]  # type: ignore[arg-type]
            med_wall = statistics.median(r[0] for r in runs)
            med_cpu = statistics.median(r[1] for r in runs)
            med_rss = statistics.median(r[2] for r in runs)
            med_merge = statistics.median(r[3] for r in runs)
            speedup = serial_wall / med_wall
            results[exec_type][w] = speedup

            print(
                f"| {exec_type:<10} | {w:>7} | {med_wall:>9.4f} | {med_cpu:>8.4f} | "
                f"{med_rss:>9.1f} | {med_merge * 1000:>10.2f} | {speedup:>7.2f}x |"
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None  # type: ignore[assignment]

    if plt is not None:
        plt.figure(figsize=(8, 5))
        xs = worker_counts
        plt.plot(xs, xs, "--", color="gray", label="Ідеальний лінійний ріст (y = x)")
        plt.plot(xs, [results["threads"][w] for w in xs], "o-", label="Threads (GIL увімкнено)")
        plt.plot(
            xs, [results["processes"][w] for w in xs], "s-", label="Processes (ProcessPoolExecutor)"
        )
        plt.xlabel("Кількість воркерів (workers)")
        plt.ylabel("Прискорення (Speedup = T_serial / T_parallel)")
        plt.title(f"Масштабування індексації findex ({get_cpu_info()})")
        plt.xticks(xs)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig("speedup_lab5.png", dpi=150)
        print("\nГрафік збережено у 'speedup_lab5.png'.")

    run_race_demo()


if __name__ == "__main__":
    main()
