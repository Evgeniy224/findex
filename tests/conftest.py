from pathlib import Path

import pytest

from findex.index import Index, build_index


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    """Малий фіксований корпус із 4 документів для детермінованих тестів."""
    docs = {
        "doc0.txt": "Python async await event loop швидка пошукова система",
        "doc1.txt": "Пошукова система використовує інвертований індекс та ранжування BM25",
        "doc2.txt": "Асинхронний python код працює через event loop без блокування",
        "doc3.txt": "Українська токенізація обробляє слова п'ять та будь-який коректно",
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    """Готовий Index, побудований на малому корпусі."""
    return build_index(sample_corpus_dir)