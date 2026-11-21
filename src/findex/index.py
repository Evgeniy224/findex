import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

ExecutorType = Literal["serial", "threads", "processes"]


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


def build_partial(paths: list[Path], start_doc_id: int = 0) -> Index:
    """
    Функція рівня модуля для воркерів (безпечна для pickle / spawn).
    Читає переданий список шляхів та призначає їм doc_id починаючи з start_doc_id.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_meta: dict[int, DocMeta] = {}

    for offset, path in enumerate(paths):
        doc_id = start_doc_id + offset
        text = path.read_text(encoding="utf-8", errors="replace")

        doc_token_counts: defaultdict[str, int] = defaultdict(int)
        doc_length = 0
        for token in tokenize(text):
            doc_token_counts[token] += 1
            doc_length += 1

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=path, length=doc_length)
        for term, tf in doc_token_counts.items():
            postings[term].append(Posting(doc_id=doc_id, tf=tf))

    sorted_postings: dict[str, list[Posting]] = {}
    for term, plist in postings.items():
        plist.sort(key=lambda p: p.doc_id)
        sorted_postings[term] = plist

    return Index(sorted_postings, doc_meta)


def merge_indexes(partials: Iterable[Index]) -> Index:
    """Об'єднує часткові індекси в один фінальний Index зі збереженням сортування постінгів."""
    merged_postings: defaultdict[str, list[Posting]] = defaultdict(list)
    merged_meta: dict[int, DocMeta] = {}

    for partial in partials:
        merged_meta.update(partial.doc_meta)
        for term, plist in partial.postings.items():
            merged_postings[term].extend(plist)

    final_postings: dict[str, list[Posting]] = {}
    for term, plist in merged_postings.items():
        plist.sort(key=lambda p: p.doc_id)
        final_postings[term] = plist

    return Index(final_postings, merged_meta)


def _split_chunks(paths: list[Path], num_chunks: int) -> list[tuple[int, list[Path]]]:
    """Ділить список шляхів на num_chunks приблизно рівних шматків зі стартовим doc_id."""
    if not paths:
        return []
    k = max(1, min(num_chunks, len(paths)))
    chunk_size = (len(paths) + k - 1) // k
    chunks: list[tuple[int, list[Path]]] = []
    for start in range(0, len(paths), chunk_size):
        chunks.append((start, paths[start : start + chunk_size]))
    return chunks


def build_index_from_docs(
    documents: Iterable[Document],
    on_doc: Callable[[Document], None] | None = None,
) -> Index:
    """Будує об'єкт Index з довільного ітератора документів (послідовно)."""
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
    executor: ExecutorType = "serial",
    workers: int = 1,
    on_chunk_done: Callable[[int], None] | None = None,
) -> Index:
    """
    Будує інвертований індекс по корпусу в директорії root або окремому файлі (.jsonl/.txt).
    Підтримує режими виконання: 'serial', 'threads', 'processes'.
    """
    jsonl_files = list(root.rglob("*.jsonl")) if root.is_dir() else []
    if root.is_file() or jsonl_files or (executor == "serial" and on_doc is not None):
        wrapped_on_doc: Callable[[Document], None] | None = on_doc
        if wrapped_on_doc is None and on_chunk_done is not None:
            chunk_cb = on_chunk_done

            def _notify_single_doc(_doc: Document) -> None:
                chunk_cb(1)

            wrapped_on_doc = _notify_single_doc
        return build_index_from_docs(iter_documents(root), on_doc=wrapped_on_doc)

    paths = sorted(root.rglob("*.txt"))
    if not paths:
        return Index({}, {})

    chunks = _split_chunks(paths, workers if executor != "serial" else 1)

    if executor == "serial":
        partials: list[Index] = []
        for start_id, chunk_paths in chunks:
            partials.append(build_partial(chunk_paths, start_id))
            if on_chunk_done is not None:
                on_chunk_done(len(chunk_paths))
        return merge_indexes(partials)

    pool_cls = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
    ordered_partials: list[Index | None] = [None] * len(chunks)

    with pool_cls(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(build_partial, chunk_paths, start_id): (idx, len(chunk_paths))
            for idx, (start_id, chunk_paths) in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx, chunk_len = future_to_idx[future]
            ordered_partials[idx] = future.result()
            if on_chunk_done is not None:
                on_chunk_done(chunk_len)

    return merge_indexes([p for p in ordered_partials if p is not None])
