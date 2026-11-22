import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from findex.index import Index


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    """Результат пошуку, відсортований за спаданням score."""

    score: float
    doc_id: int
    snippet: str = ""


class Scorer(Protocol):
    """Структурний протокол для алгоритмів ранжування."""

    def score(self, index: Index, term: str, doc_id: int) -> float: ...


class TfIdf:
    """Алгоритм ранжування TF-IDF."""

    def score(self, index: Index, term: str, doc_id: int) -> float:
        if term not in index:
            return 0.0

        tf = 0
        for posting in index[term]:
            if posting.doc_id == doc_id:
                tf = posting.tf
                break

        if tf == 0:
            return 0.0

        meta = index.get_doc_meta(doc_id)
        if meta is None or meta.length == 0:
            return 0.0

        df = index.df(term)
        idf = math.log(index.num_docs / (df + 1)) + 1.0
        return (tf / meta.length) * idf


class BM25:
    """Алгоритм ранжування Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, index: Index, term: str, doc_id: int) -> float:
        if term not in index:
            return 0.0

        tf = 0
        for posting in index[term]:
            if posting.doc_id == doc_id:
                tf = posting.tf
                break

        if tf == 0:
            return 0.0

        meta = index.get_doc_meta(doc_id)
        avg_dl = index.avg_doc_length
        if meta is None or meta.length == 0 or avg_dl == 0.0:
            return 0.0

        df = index.df(term)
        idf = math.log((index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * (1.0 - self.b + self.b * (meta.length / avg_dl))
        return idf * (numerator / denominator)


def score_terms_numpy(
    index: Index,
    terms: list[str],
    matched_doc_ids: set[int],
    scorer_name: str,
    top_k: int,
) -> list[SearchResult]:
    """Векторизоване накопичення BM25/TF-IDF із top-k через np.argpartition."""
    if not matched_doc_ids:
        return []

    scores = np.zeros(index.num_docs, dtype=np.float64)
    doc_lengths = index.numpy_doc_lengths
    avg_length = index.avg_doc_length

    for term in terms:
        doc_ids, tfs = index.numpy_postings(term)
        if doc_ids.size == 0:
            continue

        df = int(doc_ids.size)
        if scorer_name == "bm25":
            if avg_length == 0.0:
                continue
            k1 = 1.5
            b = 0.75
            idf = np.log((index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            norm = k1 * (1.0 - b + b * doc_lengths[doc_ids].astype(np.float64) / avg_length)
            term_scores = idf * tfs * (k1 + 1.0) / (tfs + norm)
        else:
            idf = math.log(index.num_docs / (df + 1)) + 1.0
            term_scores = (tfs.astype(np.float64) / doc_lengths[doc_ids]) * idf

        scores[doc_ids] += term_scores

    candidate_ids = np.fromiter(sorted(matched_doc_ids), dtype=np.int32)
    candidate_scores = scores[candidate_ids]
    limit = min(top_k, candidate_ids.size)
    if limit == 0:
        return []
    kth = limit
    if kth < candidate_ids.size:
        partition = np.argpartition(candidate_scores, -kth)[-kth:]
        threshold = float(candidate_scores[partition].min())
        candidate_ids = candidate_ids[candidate_scores >= threshold]
        candidate_scores = scores[candidate_ids]
    else:
        candidate_scores = scores[candidate_ids]

    order = np.lexsort((-candidate_ids.astype(np.int64), -candidate_scores))
    candidate_ids = candidate_ids[order[:top_k]]
    candidate_scores = candidate_scores[order[:top_k]]
    return [
        SearchResult(score=float(score), doc_id=int(doc_id))
        for doc_id, score in zip(candidate_ids, candidate_scores, strict=True)
    ]
