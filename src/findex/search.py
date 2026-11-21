import heapq
import logging
from typing import Literal

from findex.decorators import timed
from findex.index import Index
from findex.query_parser import EngineType, Node, parse_query
from findex.ranking import BM25, Scorer, SearchResult, TfIdf

logger = logging.getLogger(__name__)

ScorerName = Literal["tfidf", "bm25"]

_search_cache: dict[tuple[int, str, ScorerName, EngineType, int], list[SearchResult]] = {}


def clear_search_cache() -> None:
    """Очищує кеш пошукових запитів."""
    _search_cache.clear()


def _extract_terms(node: Node) -> list[str]:
    from findex.query_parser import And, Or, Phrase, Term

    if isinstance(node, Term):
        return [node.term]
    if isinstance(node, Phrase):
        return list(node.terms)
    if isinstance(node, And | Or):
        return _extract_terms(node.left) + _extract_terms(node.right)
    return []


@timed
def search(
    index: Index,
    query: str,
    scorer_name: ScorerName = "tfidf",
    engine: EngineType = "set",
    top_k: int = 10,
    use_cache: bool = True,
) -> list[SearchResult]:
    """Пошук по індексу з підтримкою булевої логіки, вибору рушія та скорера."""
    cache_key = (id(index), query, scorer_name, engine, top_k)

    if use_cache and cache_key in _search_cache:
        logger.debug("Cache hit для запиту '%s'", query)
        return _search_cache[cache_key]

    ast = parse_query(query)
    matched_docs = ast.evaluate(index, engine=engine)
    if not matched_docs:
        return []

    terms = _extract_terms(ast)
    scorer: Scorer = BM25() if scorer_name == "bm25" else TfIdf()

    scored_results: list[SearchResult] = []
    for doc_id in matched_docs:
        total_score = sum(scorer.score(index, term, doc_id) for term in terms)
        scored_results.append(SearchResult(score=total_score, doc_id=doc_id))

    top_results = heapq.nlargest(top_k, scored_results)

    final_results: list[SearchResult] = []
    for res in top_results:
        snippet = generate_snippet(index, res.doc_id, terms)
        final_results.append(SearchResult(score=res.score, doc_id=res.doc_id, snippet=snippet))

    if use_cache:
        _search_cache[cache_key] = final_results
    return final_results


def generate_snippet(index: Index, doc_id: int, terms: list[str], window: int = 80) -> str:
    """Генерує уривок тексту навколо найкращого входження термінів із підсвіткою."""
    meta = index.get_doc_meta(doc_id)
    if meta is None:
        return ""

    try:
        text = meta.path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if not text:
        return ""

    text_lower = text.casefold()
    best_pos = -1
    best_term = ""
    for term in terms:
        pos = text_lower.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_term = term

    if best_pos == -1:
        preview = text[: window * 2].replace("\n", " ")
        return f"{preview}..." if len(text) > window * 2 else preview

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + len(best_term) + window)
    snippet = text[start:end].replace("\n", " ")

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    for term in dict.fromkeys(terms):
        if term:
            snippet = snippet.replace(term, f"*{term}*")
            snippet = snippet.replace(term.capitalize(), f"*{term.capitalize()}*")

    return snippet
