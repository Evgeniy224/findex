import asyncio
import json
import logging
import resource
import sys
from pathlib import Path
from typing import Annotated, Literal, TextIO

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import DEFAULT_USER_AGENT, FetchResult
from findex.index import ExecutorType, build_index
from findex.query_parser import QuerySyntaxError
from findex.search import search
from findex.stats import compute_stats
from findex.store import open_index, save_index

app = typer.Typer(
    name="findex",
    help="Пошукова система findex: краулінг, індексація, статистика та ранжований пошук.",
    no_args_is_help=True,
    add_completion=False,
)

stdout_console = Console(stderr=False)
stderr_console = Console(stderr=True)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=stderr_console, show_path=False, rich_tracebacks=False)],
        force=True,
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    stderr_console.print(f"[bold red]Помилка:[/bold red] {message}")
    return typer.Exit(code=code)


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _write_and_flush(file_handle: TextIO, lines: list[str]) -> None:
    file_handle.writelines(lines)
    file_handle.flush()


def _build_crawl_table(stats: CrawlStats, max_pages: int, out_path: Path) -> Table:
    table = Table(title=f"findex crawl → {out_path}")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значення", justify="right", style="bold green")
    table.add_row("Зібрано сторінок", f"{stats.fetched} / {max_pages}")
    table.add_row("Швидкість (pages/s)", f"{stats.pages_per_sec:.2f}")
    table.add_row("Зараз у польоті (in-flight)", str(stats.in_flight))
    table.add_row("Помилок", f"[red]{stats.errors}[/red]" if stats.errors else "0")
    table.add_row("Глибина черги (frontier)", str(stats.queue_depth))
    table.add_row("Час (s)", f"{stats.elapsed:.2f}")
    return table


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Рівень деталізації логів (-v для INFO, -vv для DEBUG).",
        ),
    ] = 0,
) -> None:
    """Глобальні параметри CLI findex."""
    _configure_logging(verbose)


@app.command("crawl")
def cmd_crawl(
    seed_url: Annotated[str, typer.Argument(help="Початковий URL для обходу.")],
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", "-m", min=1, help="Максимальна кількість сторінок."),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, help="Кількість одночасних воркерів."),
    ] = 10,
    per_host: Annotated[
        int,
        typer.Option("--per-host", min=1, help="Максимум одночасних запитів на один хост."),
    ] = 2,
    min_delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Мінімальна пауза (с) між запитами на хост."),
    ] = 0.2,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Вихідний JSONL-файл зі сторінками."),
    ] = Path("data/crawl.jsonl"),
    log_file: Annotated[
        Path,
        typer.Option("--log", help="Файл журналу статусів по кожному URL."),
    ] = Path("crawl.log"),
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Увімкнути asyncio debug mode для виявлення блокувань."),
    ] = False,
) -> None:
    """Асинхронно обійти веб-сторінки та зберегти корпус у потоковий JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    stats = CrawlStats()
    records: list[FetchResult] = []

    async def _run_crawl() -> None:
        with (
            out.open("w", encoding="utf-8") as jsonl_fh,
            log_file.open("w", encoding="utf-8") as log_fh,
            Live(
                _build_crawl_table(stats, max_pages, out),
                console=stderr_console,
                refresh_per_second=8,
            ) as live,
        ):
            await asyncio.to_thread(
                _write_and_flush,
                log_fh,
                [
                    f"# User-Agent: {DEFAULT_USER_AGENT}\n",
                    "status\tbytes\telapsed_s\tattempts\turl\terror\n",
                ],
            )
            flushed = 0

            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=min_delay,
                stats=stats,
                log_records=records,
            ):
                line = json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )
                await asyncio.to_thread(_write_and_flush, jsonl_fh, [line + "\n"])

                log_lines: list[str] = []
                while flushed < len(records):
                    r = records[flushed]
                    log_lines.append(
                        f"{r.status}\t{r.num_bytes}\t{r.elapsed:.4f}\t"
                        f"{r.attempts}\t{r.url}\t{r.error or '-'}\n"
                    )
                    flushed += 1
                if log_lines:
                    await asyncio.to_thread(_write_and_flush, log_fh, log_lines)

                live.update(_build_crawl_table(stats, max_pages, out))

            log_lines = []
            while flushed < len(records):
                r = records[flushed]
                log_lines.append(
                    f"{r.status}\t{r.num_bytes}\t{r.elapsed:.4f}\t"
                    f"{r.attempts}\t{r.url}\t{r.error or '-'}\n"
                )
                flushed += 1
            if log_lines:
                await asyncio.to_thread(_write_and_flush, log_fh, log_lines)
            live.update(_build_crawl_table(stats, max_pages, out))

    asyncio.run(_run_crawl(), debug=debug)

    stderr_console.print(
        f"[green]Завершено![/green] Сторінок: {stats.fetched}, час: {stats.elapsed:.2f}s "
        f"({stats.pages_per_sec:.2f} pages/s), помилок: {stats.errors}, "
        f"пік RSS: {_rss_mb():.1f} MB."
    )


@app.command("index")
def cmd_index(
    root: Annotated[Path, typer.Argument(help="Директорія з .txt документами або .jsonl файл.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Шлях до вихідного JSON-файлу індексу."),
    ] = Path("index.json"),
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Кількість воркерів для індексації."),
    ] = 1,
    executor: Annotated[
        ExecutorType,
        typer.Option(
            "--executor",
            "-e",
            help="Тип виконавця: serial, threads або processes.",
        ),
    ] = "serial",
) -> None:
    """Побудувати інвертований індекс із корпусу документів."""
    if not root.exists():
        raise _fail(f"Шлях корпусу не знайдено: '{root}'")

    if root.is_file():
        total_files: int | None = None
    else:
        txt_files = list(root.rglob("*.txt")) + list(root.rglob("*.jsonl"))
        total_files = len(txt_files)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=stderr_console,
    ) as progress:
        task = progress.add_task(
            f"Індексація ({executor}, workers={workers})...",
            total=total_files,
        )
        index_obj = build_index(
            root,
            executor=executor,
            workers=workers,
            on_chunk_done=lambda count: progress.advance(task, count),
        )

    save_index(index_obj, out)
    stderr_console.print(
        f"[green]Готово![/green] Проіндексовано документів: {index_obj.num_docs}, "
        f"термів: {len(index_obj)}. Збережено у '{out}'."
    )


@app.command("search")
def cmd_search(
    index_file: Annotated[Path, typer.Argument(help="Шлях до JSON-файлу індексу.")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит (AND, OR, NOT, дужки, фрази).")],
    scorer: Annotated[
        Literal["tfidf", "bm25"],
        typer.Option("--scorer", "-s", help="Алгоритм ранжування: tfidf або bm25."),
    ] = "tfidf",
    engine: Annotated[
        Literal["set", "merge"],
        typer.Option("--engine", "-e", help="Рушій перетину списків: set або merge."),
    ] = "set",
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", min=1, help="Максимальна кількість результатів."),
    ] = 10,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Вивести результати у форматі JSON у stdout."),
    ] = False,
) -> None:
    """Виконати пошуковий запит за побудованим індексом."""
    if not index_file.exists() or not index_file.is_file():
        raise _fail(f"Файл індексу не знайдено: '{index_file}'")

    try:
        with open_index(index_file) as index_obj:
            results = search(
                index_obj,
                query=query,
                scorer_name=scorer,
                engine=engine,
                top_k=top_k,
            )

            if as_json:
                payload = [
                    {
                        "rank": rank,
                        "doc_id": res.doc_id,
                        "path": (
                            str(meta.path) if (meta := index_obj.get_doc_meta(res.doc_id)) else ""
                        ),
                        "score": round(res.score, 6),
                        "snippet": res.snippet,
                    }
                    for rank, res in enumerate(results, start=1)
                ]
                typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
                return

            if not results:
                stderr_console.print(f"[yellow]За запитом '{query}' нічого не знайдено.[/yellow]")
                return

            table = Table(title=f"Результати пошуку: '{query}' ({scorer.upper()}, engine={engine})")
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column("Doc ID", justify="right", style="magenta")
            table.add_column("Файл", style="green")
            table.add_column("Score", justify="right", style="bold yellow")
            table.add_column("Сніпет", style="white")

            for rank, res in enumerate(results, start=1):
                meta = index_obj.get_doc_meta(res.doc_id)
                filename = meta.path.name if meta is not None else f"doc_{res.doc_id}"
                table.add_row(
                    str(rank),
                    str(res.doc_id),
                    filename,
                    f"{res.score:.4f}",
                    res.snippet,
                )

            stdout_console.print(table)

    except QuerySyntaxError as exc:
        raise _fail(f"Некоректний пошуковий запит: {exc}") from None
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise _fail(f"Пошкоджений файл індексу '{index_file}': {exc}") from None


@app.command("stats")
def cmd_stats(
    root: Annotated[Path, typer.Argument(help="Директорія з .txt документами.")],
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", min=1, help="Обмежити кількість документів."),
    ] = None,
) -> None:
    """Показати статистику текстового корпусу."""
    if not root.exists() or not root.is_dir():
        raise _fail(f"Директорію корпусу не знайдено: '{root}'")

    stats_data = compute_stats(root, limit=limit)

    summary_table = Table(title=f"Статистика корпусу '{root}'")
    summary_table.add_column("Метрика", style="cyan")
    summary_table.add_column("Значення", justify="right", style="bold green")

    summary_table.add_row("Документів", str(stats_data["documents"]))
    summary_table.add_row("Всього токенів", str(stats_data["tokens"]))
    summary_table.add_row("Унікальних термів (словник)", str(stats_data["vocabulary"]))
    summary_table.add_row("Час обробки (сек)", f"{stats_data['elapsed']:.4f}")
    summary_table.add_row("Пікова пам'ять (МБ)", f"{stats_data['peak_memory_mb']:.4f}")

    stdout_console.print(summary_table)

    if stats_data["top_terms"]:
        top_table = Table(title="Топ-10 найчастіших термів")
        top_table.add_column("Терм", style="yellow")
        top_table.add_column("Частота", justify="right", style="cyan")
        for term, count in stats_data["top_terms"][:10]:
            top_table.add_row(term, str(count))
        stdout_console.print(top_table)


if __name__ == "__main__":
    app()
