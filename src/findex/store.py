import pickle
import json
from pathlib import Path
from findex.index import Posting, DocMeta


def save_index(postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta], path: Path):
    """Зберігає індекс у файл через pickle."""
    with open(path, "wb") as f:
        pickle.dump((postings, doc_meta), f)


def load_index(path: Path) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Завантажує індекс з файлу pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_index_json(postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta], path: Path):
    """Зберігає індекс у JSON (безпечно, але повільніше)."""
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


def load_index_json(path: Path) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """Завантажує індекс з JSON."""
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