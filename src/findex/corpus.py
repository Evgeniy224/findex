import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple, cast

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: int
    path: Path
    text: str


def _iter_jsonl_file(jsonl_path: Path, start_id: int) -> Iterator[Document]:
    """Ліниво читає документи рядок за рядком із JSONL-файлу."""
    doc_id = start_id
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_obj: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Некоректний JSON у %s:%d: %s", jsonl_path, line_no, exc)
                    continue

                if not isinstance(raw_obj, dict):
                    continue

                record = cast(dict[str, Any], raw_obj)
                url_or_path = str(record.get("url") or f"{jsonl_path}#{line_no}")
                title = str(record.get("title") or "").strip()
                body = str(record.get("text") or "").strip()
                full_text = f"{title}\n{body}".strip() if title else body

                yield Document(doc_id=doc_id, path=Path(url_or_path), text=full_text)
                doc_id += 1
    except OSError as exc:
        logger.warning("Не вдалося прочитати JSONL %s: %s", jsonl_path, exc)


def iter_documents(root: Path) -> Iterator[Document]:
    """
    Лінивий генератор документів.
    Читає .jsonl файл, або всі .txt та .jsonl файли в директорії root по одному.
    """
    if root.is_file():
        if root.suffix.lower() == ".jsonl":
            yield from _iter_jsonl_file(root, start_id=0)
        elif root.suffix.lower() == ".txt":
            try:
                text = root.read_text(encoding="utf-8", errors="replace")
                yield Document(0, root, text)
            except OSError as exc:
                logger.warning("Не вдалося прочитати %s: %s", root, exc)
        return

    doc_id = 0
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield Document(doc_id, path, text)
            doc_id += 1
        except OSError as exc:
            logger.warning("Не вдалося прочитати %s: %s", path, exc)
            continue

    for jsonl_path in sorted(root.rglob("*.jsonl")):
        for doc in _iter_jsonl_file(jsonl_path, start_id=doc_id):
            yield doc
            doc_id = doc.doc_id + 1
