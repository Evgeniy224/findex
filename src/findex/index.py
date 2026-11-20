from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: Path
    length: int


class Index(Mapping[str, list[Posting]]):
    """Незмінний інвертований індекс (term -> postings) з метаданими документів."""

    def __init__(self, postings: dict[str, list[Posting]], doc_meta: dict[int, DocMeta]) -> None:
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return self._postings == other._postings and self._doc_meta == other._doc_meta

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    @property
    def doc_meta(self) -> dict[int, DocMeta]:
        return self._doc_meta

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
        """Document frequency — кількість документів, що містять термін."""
        return len(self._postings.get(term, []))

    def get_doc_meta(self, doc_id: int) -> DocMeta | None:
        return self._doc_meta.get(doc_id)

    def all_doc_ids(self) -> set[int]:
        return set(self._doc_meta.keys())


def intersect_postings(left: list[Posting], right: list[Posting]) -> list[int]:
    """Двовказівниковий перетин (AND) двох відсортованих списків постінгів."""
    i, j = 0, 0
    result: list[int] = []
    while i < len(left) and j < len(right):
        if left[i].doc_id == right[j].doc_id:
            result.append(left[i].doc_id)
            i += 1
            j += 1
        elif left[i].doc_id < right[j].doc_id:
            i += 1
        else:
            j += 1
    return result


def union_postings(left: list[Posting], right: list[Posting]) -> list[int]:
    """Двовказівникове об'єднання (OR) двох відсортованих списків постінгів."""
    i, j = 0, 0
    result: list[int] = []
    while i < len(left) and j < len(right):
        if left[i].doc_id == right[j].doc_id:
            result.append(left[i].doc_id)
            i += 1
            j += 1
        elif left[i].doc_id < right[j].doc_id:
            result.append(left[i].doc_id)
            i += 1
        else:
            result.append(right[j].doc_id)
            j += 1
    while i < len(left):
        result.append(left[i].doc_id)
        i += 1
    while j < len(right):
        result.append(right[j].doc_id)
        j += 1
    return result


def build_index_from_docs(
    documents: Iterable[Document],
    on_doc: Callable[[Document], None] | None = None,
) -> Index:
    """Будує об'єкт Index з довільного ітератора документів."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for doc in documents:
        doc_token_counts: defaultdict[str, int] = defaultdict(int)
        doc_length = 0
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
            doc_length += 1

        doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, path=doc.path, length=doc_length)

        for term, tf in doc_token_counts.items():
            postings[term].append(Posting(doc_id=doc.doc_id, tf=tf))

        if on_doc is not None:
            on_doc(doc)

    sorted_postings: dict[str, list[Posting]] = {}
    for term, plist in postings.items():
        plist.sort(key=lambda p: p.doc_id)
        sorted_postings[term] = plist

    return Index(sorted_postings, doc_meta)


def build_index(
    root: Path,
    on_doc: Callable[[Document], None] | None = None,
) -> Index:
    """Будує інвертований індекс за один прохід по корпусу в директорії root."""
    return build_index_from_docs(iter_documents(root), on_doc=on_doc)