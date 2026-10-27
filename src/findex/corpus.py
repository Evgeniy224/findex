from pathlib import Path
from typing import Iterator, NamedTuple
import logging

logging.basicConfig(level=logging.INFO)

class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str

def iter_documents(root: Path) -> Iterator[Document]:
    """Читає .txt файли з папки root по одному."""
    doc_id = 0
    # rglob шукає файли .txt у всіх підпапках
    for path in root.rglob("*.txt"):
        try:
            # encoding="utf-8" обов'язковий для коректного читання
            # errors="replace" замінить биті символи на �, щоб код не впав
            text = path.read_text(encoding="utf-8", errors="replace")
            yield Document(doc_id, path, text)
            doc_id += 1
        except Exception as e:
            # Якщо файл не читається, просто попереджаємо і йдемо далі
            logging.warning(f"Не вдалося прочитати {path}: {e}")
            continue