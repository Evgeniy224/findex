import sys
from pathlib import Path

# Додаємо папку src у шлях пошуку модулів
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Iterator
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # term frequency - скільки разів термін зустрічається в документі


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: Path
    length: int  # кількість токенів у документі


class Index(Mapping):
    """
    Клас, що представляє інвертований індекс.
    Поводиться як незмінний словник (term -> postings),
    але також надає доступ до метаданих документів.
    """
    def __init__(self, postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta]):
        self._postings = postings
        self._doc_meta = doc_meta

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self._postings)}, docs={len(self._doc_meta)})"

    @property
    def num_docs(self) -> int:
        return len(self._doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_meta:
            return 0.0
        total = sum(meta.length for meta in self._doc_meta.values())
        return total / len(self._doc_meta)

    def df(self, term: str) -> int:
        """Document frequency - кількість документів, що містять термін."""
        return len(self._postings.get(term, []))

    def get_doc_meta(self, doc_id: int) -> DocMeta | None:
        return self._doc_meta.get(doc_id)

    def all_doc_ids(self) -> set[int]:
        return set(self._doc_meta.keys())


def build_index(root: Path) -> tuple[dict[str, list[Posting]], dict[int, DocMeta]]:
    """
    Будує інвертований індекс за один прохід по корпусу.
    Повертає (postings, doc_meta).
    """
    postings = defaultdict(list)
    doc_meta = {}

    for doc in iter_documents(root):
        doc_token_counts = defaultdict(int)
        doc_length = 0
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
            doc_length += 1

        doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, path=doc.path, length=doc_length)

        for term, tf in doc_token_counts.items():
            postings[term].append(Posting(doc_id=doc.doc_id, tf=tf))

    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)

    return dict(postings), doc_meta


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc
    from findex.store import save_index

    parser = argparse.ArgumentParser(description="Побудова та збереження інвертованого індексу")
    parser.add_argument("root", type=Path, help="Шлях до папки з корпусом")
    parser.add_argument("--out", type=Path, default=Path("index.json"), help="Файл для збереження індексу")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    postings, doc_meta = build_index(args.root)

    # Тепер save_index завжди зберігає в JSON
    save_index(postings, doc_meta, args.out)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано та збережено у {args.out}")
    print(f"Термів: {len(postings)}, Документів: {len(doc_meta)}")
    print(f"Час: {elapsed:.4f} сек, Пікова пам'ять: {peak / (1024 * 1024):.4f} МБ")