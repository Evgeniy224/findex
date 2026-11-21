import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout


def test_cli_index_and_search_json(sample_corpus_dir: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "cli_index.json"

    res_index = runner.invoke(app, ["index", str(sample_corpus_dir), "--out", str(index_path)])
    assert res_index.exit_code == 0
    assert index_path.exists()

    res_search = runner.invoke(
        app,
        ["search", str(index_path), "python AND async", "--scorer", "bm25", "--json"],
    )
    assert res_search.exit_code == 0
    raw_data = json.loads(res_search.stdout)
    assert isinstance(raw_data, list)
    data = cast(list[dict[str, Any]], raw_data)
    assert len(data) == 1
    assert data[0]["doc_id"] == 0


def test_cli_search_rich_table_and_verbose(sample_corpus_dir: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "cli_index.json"
    runner.invoke(app, ["index", str(sample_corpus_dir), "--out", str(index_path)])

    res_table = runner.invoke(app, ["-vv", "search", str(index_path), "пошукова"])
    assert res_table.exit_code == 0
    assert "Результати пошуку" in res_table.stdout


def test_cli_stats(sample_corpus_dir: Path) -> None:
    res = runner.invoke(app, ["stats", str(sample_corpus_dir)])
    assert res.exit_code == 0
    assert "Статистика корпусу" in res.stdout


def test_cli_missing_index_fails_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.json"
    res = runner.invoke(app, ["search", str(missing), "python"])
    assert res.exit_code != 0
    assert "Помилка:" in res.stderr


def test_cli_broken_query_fails_cleanly(sample_corpus_dir: Path, tmp_path: Path) -> None:
    index_path = tmp_path / "cli_index.json"
    runner.invoke(app, ["index", str(sample_corpus_dir), "--out", str(index_path)])

    res = runner.invoke(app, ["search", str(index_path), "(python AND"])
    assert res.exit_code != 0
    assert "Помилка:" in res.stderr
