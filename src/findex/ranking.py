import math
from dataclasses import dataclass
from typing import Protocol

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
