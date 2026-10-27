import json
import pickle
from pathlib import Path
from contextlib import contextmanager
from findex.index import Posting, DocMeta, Index


def save_index(postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta], path: Path):
    """
    Зберігає індекс у JSON (безпечно, читабельно, без проблем з pickle).
    """
    data = {
        "postings": {
            term: [{"doc_id": p.doc_id, "tf": p.tf} for p in plist]
            for term, plist in postings.items()
        },
        "doc_meta": {
            str(doc_id): {"path": str(meta.path), "length": meta.length}
            for doc_id, meta in doc_meta.items()
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_index(path: Path) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """
    Завантажує індекс з JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    postings = {
        term: [Posting(doc_id=p["doc_id"], tf=p["tf"]) for p in plist]
        for term, plist in data["postings"].items()
    }
    doc_meta = {
        int(doc_id): DocMeta(doc_id=int(doc_id), path=Path(meta["path"]), length=meta["length"])
        for doc_id, meta in data["doc_meta"].items()
    }
    return postings, doc_meta


@contextmanager
def open_index(path: Path):
    """
    Контекстний менеджер для завантаження індексу.
    """
    postings, doc_meta = load_index(path)
    index_obj = Index(postings, doc_meta)
    try:
        yield index_obj
    finally:
        pass