import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from findex.index import DocMeta, Index, Posting


class _PostingDict(TypedDict):
    doc_id: int
    tf: int


class _DocMetaDict(TypedDict):
    path: str
    length: int


class _SerializedIndex(TypedDict):
    postings: dict[str, list[_PostingDict]]
    doc_meta: dict[str, _DocMetaDict]


def save_index(index: Index, path: Path) -> None:
    """Зберігає індекс у безпечному форматі JSON."""
    payload: _SerializedIndex = {
        "postings": {
            term: [{"doc_id": p.doc_id, "tf": p.tf} for p in plist]
            for term, plist in index.postings.items()
        },
        "doc_meta": {
            str(doc_id): {"path": str(meta.path), "length": meta.length}
            for doc_id, meta in index.doc_meta.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_index(path: Path) -> Index:
    """Завантажує індекс з JSON-файлу."""
    with path.open("r", encoding="utf-8") as f:
        raw: _SerializedIndex = json.load(f)

    postings: dict[str, list[Posting]] = {
        term: [Posting(doc_id=int(p["doc_id"]), tf=int(p["tf"])) for p in plist]
        for term, plist in raw["postings"].items()
    }
    doc_meta: dict[int, DocMeta] = {
        int(doc_id): DocMeta(
            doc_id=int(doc_id),
            path=Path(meta["path"]),
            length=int(meta["length"]),
        )
        for doc_id, meta in raw["doc_meta"].items()
    }
    return Index(postings, doc_meta)


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """Контекстний менеджер для завантаження індексу."""
    index_obj = load_index(path)
    yield index_obj