import logging
from pathlib import Path
from typing import Literal

from findex.decorators import timed
from findex.index import Index
from findex.query_parser import EngineType, Node, QuerySyntaxError, parse_query
from findex.ranking import SearchResult, score_terms_numpy
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

ScorerName = Literal["tfidf", "bm25"]
SearchMode = Literal["keyword", "semantic", "hybrid"]

_search_cache: dict[
    tuple[int, str, ScorerName, EngineType, SearchMode, int],
    list[SearchResult],
] = {}


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
    mode: SearchMode = "keyword",
    index_path: Path | None = None,
    embeddings_model: str = "all-MiniLM-L6-v2",
) -> list[SearchResult]:
    """Пошук по індексу з підтримкою булевої логіки, вибору рушія та скорера."""
    cache_key = (id(index), query, scorer_name, engine, mode, top_k)

    if use_cache and cache_key in _search_cache:
        logger.debug("Cache hit для запиту '%s'", query)
        return _search_cache[cache_key]

    terms = list(tokenize(query))
    matched_docs: set[int]
    if mode == "keyword":
        ast = parse_query(query)
        matched_docs = ast.evaluate(index, engine=engine)
        terms = _extract_terms(ast)
    else:
        try:
            ast = parse_query(query)
            matched_docs = ast.evaluate(index, engine=engine)
            terms = _extract_terms(ast)
        except QuerySyntaxError:
            matched_docs = set()
            for term in terms:
                matched_docs.update(posting.doc_id for posting in index.postings.get(term, []))

    if mode != "keyword" and index_path is not None:
        from findex.embeddings import reciprocal_rank_fusion, semantic_search

        semantic = semantic_search(
            index_path,
            query,
            top_k,
            model_name=embeddings_model,
        )
        if mode == "semantic":
            top_results = semantic
        else:
            keyword = (
                score_terms_numpy(index, terms, matched_docs, scorer_name, top_k)
                if matched_docs
                else []
            )
            top_results = reciprocal_rank_fusion([keyword, semantic], top_k)
    else:
        top_results = score_terms_numpy(
            index,
            terms,
            matched_docs,
            scorer_name,
            top_k,
        )

    if not matched_docs:
        if mode == "keyword":
            return []
        terms = []

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
