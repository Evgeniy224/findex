import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.index import ExecutorType, build_index
from findex.query_parser import QuerySyntaxError
from findex.search import search
from findex.stats import compute_stats
from findex.store import open_index, save_index

app = typer.Typer(
    name="findex",
    help="Пошукова система findex: індексація, статистика та ранжований пошук.",
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


@app.command("index")
def cmd_index(
    root: Annotated[Path, typer.Argument(help="Директорія з .txt документами.")],
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
    if not root.exists() or not root.is_dir():
        raise _fail(f"Директорію корпусу не знайдено: '{root}'")

    txt_files = list(root.rglob("*.txt"))
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
