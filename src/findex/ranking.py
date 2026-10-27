import math
from dataclasses import dataclass
from findex.index import Index


@dataclass(order=True)
class SearchResult:
    """Результат пошуку, який сортується за score (спадання)."""
    score: float
    doc_id: int
    snippet: str = ""  # Поки що порожній, додамо пізніше


class Scorer:
    """Базовий протокол для всіх алгоритмів ранжування."""

    def score(self, index: Index, term: str, doc_id: int) -> float:
        raise NotImplementedError


class TfIdf(Scorer):
    """Алгоритм TF-IDF."""

    def score(self, index: Index, term: str, doc_id: int) -> float:
        if term not in index:
            return 0.0

        # Знаходимо tf (term frequency) для цього документа
        tf = 0
        for posting in index[term]:
            if posting.doc_id == doc_id:
                tf = posting.tf
                break

        if tf == 0:
            return 0.0

        # IDF (inverse document frequency)
        df = index.df(term)
        idf = math.log(index.num_docs / (df + 1)) + 1  # +1 для згладжування

        # Нормалізуємо на довжину документа
        doc_len = index.get_doc_meta(doc_id).length
        if doc_len == 0:
            return 0.0

        return (tf / doc_len) * idf


class BM25(Scorer):
    """Алгоритм BM25 (Okapi BM25)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, index: Index, term: str, doc_id: int) -> float:
        if term not in index:
            return 0.0

        # Знаходимо tf
        tf = 0
        for posting in index[term]:
            if posting.doc_id == doc_id:
                tf = posting.tf
                break

        if tf == 0:
            return 0.0

        # IDF
        df = index.df(term)
        idf = math.log((index.num_docs - df + 0.5) / (df + 0.5) + 1)

        # Довжина документа
        doc_len = index.get_doc_meta(doc_id).length
        avg_dl = index.avg_doc_length

        # Формула BM25
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / avg_dl))

        return idf * (numerator / denominator)