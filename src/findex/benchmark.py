import logging
import tracemalloc
from array import array
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass
class PostingPlain:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class PostingSlots:
    doc_id: int
    tf: int


def build_index_plain(root: Path) -> dict[str, list[PostingPlain]]:
    """Будує індекс зі звичайними dataclass."""
    postings: defaultdict[str, list[PostingPlain]] = defaultdict(list)
    for doc in iter_documents(root):
        doc_token_counts: defaultdict[str, int] = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            postings[term].append(PostingPlain(doc_id=doc.doc_id, tf=tf))
    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)
    return dict(postings)


def build_index_slots(root: Path) -> dict[str, list[PostingSlots]]:
    """Будує індекс з dataclass slots=True."""
    postings: defaultdict[str, list[PostingSlots]] = defaultdict(list)
    for doc in iter_documents(root):
        doc_token_counts: defaultdict[str, int] = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            postings[term].append(PostingSlots(doc_id=doc.doc_id, tf=tf))
    for term in postings:
        postings[term].sort(key=lambda p: p.doc_id)
    return dict(postings)


def build_index_array(root: Path) -> dict[str, tuple[array[int], array[int]]]:
    """Будує індекс з використанням array('I')."""
    postings: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    for doc in iter_documents(root):
        doc_token_counts: defaultdict[str, int] = defaultdict(int)
        for token in tokenize(doc.text):
            doc_token_counts[token] += 1
        for term, tf in doc_token_counts.items():
            doc_ids, tfs = postings[term]
            doc_ids.append(doc.doc_id)
            tfs.append(tf)

    result: dict[str, tuple[array[int], array[int]]] = {}
    for term, (doc_ids, tfs) in postings.items():
        pairs = sorted(zip(doc_ids, tfs, strict=True))
        new_doc_ids = array("I", (p[0] for p in pairs))
        new_tfs = array("I", (p[1] for p in pairs))
        result[term] = (new_doc_ids, new_tfs)
    return result


def measure_memory[T](build_func: Callable[[Path], T], root: Path) -> tuple[float, T]:
    """Вимірює пікову пам'ять для функції побудови у МБ."""
    tracemalloc.start()
    index = build_func(root)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024), index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    corpus_root = Path("data")
    logger.info("Вимірювання пам'яті для різних варіантів Posting...")
    mem_plain, _ = measure_memory(build_index_plain, corpus_root)
    logger.info("1. Plain dataclass: %.4f МБ", mem_plain)
    mem_slots, _ = measure_memory(build_index_slots, corpus_root)
    logger.info("2. Slots dataclass: %.4f МБ", mem_slots)
    mem_array, _ = measure_memory(build_index_array, corpus_root)
    logger.info("3. Array('I'):      %.4f МБ", mem_array)
