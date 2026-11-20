import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Читає .txt файли з папки root по одному (лінивий генератор)."""
    doc_id = 0
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield Document(doc_id, path, text)
            doc_id += 1
        except OSError as exc:
            logger.warning("Не вдалося прочитати %s: %s", path, exc)
            continue