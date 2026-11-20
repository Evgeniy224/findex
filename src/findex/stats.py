import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import TypedDict

from findex.corpus import iter_documents
from findex.tokenize import tokenize


class CorpusStats(TypedDict):
    documents: int
    tokens: int
    vocabulary: int
    top_terms: list[tuple[str, int]]
    elapsed: float
    peak_memory_mb: float


def compute_stats(root: Path, limit: int | None = None) -> CorpusStats:
    """Лінивий підрахунок статистики корпусу."""
    tracemalloc.start()
    start_time = time.perf_counter()

    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    doc_count = 0
    total_tokens = 0
    term_counts: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for token in tokenize(doc.text):
            term_counts[token] += 1
            total_tokens += 1

    elapsed = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "documents": doc_count,
        "tokens": total_tokens,
        "vocabulary": len(term_counts),
        "top_terms": term_counts.most_common(20),
        "elapsed": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
    }