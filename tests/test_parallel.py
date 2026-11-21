from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_index, build_partial, merge_indexes
from findex.store import load_index

runner = CliRunner()


def _create_sample_corpus(tmp_path: Path, count: int = 12) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(count):
        (corpus_dir / f"doc_{i:02d}.txt").write_text(
            f"Документ номер {i}. Python конкурентність потоки процеси індекс слово_{i % 4}.",
            encoding="utf-8",
        )
    return corpus_dir


def test_build_partial_and_merge_matches_serial_build_index(tmp_path: Path) -> None:
    """merge([build_partial(усі шляхи)]) дає той самий індекс, що й послідовний build_index."""
    corpus_dir = _create_sample_corpus(tmp_path, count=10)
    paths = sorted(corpus_dir.rglob("*.txt"))

    expected = build_index(corpus_dir, executor="serial", workers=1)
    actual = merge_indexes([build_partial(paths, start_doc_id=0)])

    assert actual == expected
    for plist in actual.postings.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_executors_produce_identical_index(
    tmp_path: Path,
    executor: str,
    workers: int,
) -> None:
    """Усі три режими (serial, threads, processes) дають однаковий результат."""
    corpus_dir = _create_sample_corpus(tmp_path, count=15)
    baseline = build_index(corpus_dir, executor="serial", workers=1)

    result = build_index(
        corpus_dir,
        executor=executor,  # type: ignore[arg-type]
        workers=workers,
    )

    assert result == baseline
    assert result.num_docs == 15


def test_worker_exception_propagates_without_hanging() -> None:
    """Якщо воркер падає (неіснуючий файл), помилка прокидається без зависання."""
    missing_paths = [
        Path("/nonexistent_dir_xyz/missing_1.txt"),
        Path("/nonexistent_dir_xyz/missing_2.txt"),
    ]
    with pytest.raises(FileNotFoundError):
        build_partial(missing_paths, start_doc_id=0)


def test_cli_index_with_workers_and_executor(tmp_path: Path) -> None:
    """CLI підтримує прапорці --workers та --executor."""
    corpus_dir = _create_sample_corpus(tmp_path, count=8)
    out_file = tmp_path / "out_index.json"

    res = runner.invoke(
        app,
        [
            "index",
            str(corpus_dir),
            "--workers",
            "2",
            "--executor",
            "processes",
            "--out",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()

    loaded = load_index(out_file)
    assert loaded.num_docs == 8
