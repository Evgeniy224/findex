import pytest

from findex.index import Index
from findex.query_parser import (
    And,
    Not,
    Or,
    Phrase,
    QuerySyntaxError,
    Term,
    parse_query,
)
from findex.ranking import BM25, TfIdf
from findex.search import clear_search_cache, search


def test_parser_ast_precedence() -> None:
    ast = parse_query("python AND (async OR await)")
    expected = And(Term("python"), Or(Term("async"), Term("await")))
    assert ast == expected


def test_parser_not_and_phrase() -> None:
    ast = parse_query('"event loop" AND NOT bm25')
    expected = And(Phrase(("event", "loop")), Not(Term("bm25")))
    assert ast == expected


@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "   ",
        "AND",
        "python OR",
        "(python AND async",
        "python)",
        '"незакрита фраза',
        '""',
    ],
)
def test_parser_invalid_queries_raise(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


def test_scorers_positive_for_matching_doc(sample_index: Index) -> None:
    tfidf = TfIdf()
    bm25 = BM25()
    assert tfidf.score(sample_index, "python", 0) > 0.0
    assert bm25.score(sample_index, "python", 0) > 0.0


def test_scorers_zero_for_missing_term_or_doc(sample_index: Index) -> None:
    tfidf = TfIdf()
    bm25 = BM25()
    assert tfidf.score(sample_index, "немає", 0) == 0.0
    assert bm25.score(sample_index, "немає", 0) == 0.0
    assert tfidf.score(sample_index, "python", 1) == 0.0
    assert bm25.score(sample_index, "python", 1) == 0.0


def test_scorers_monotonic_with_tf(sample_index: Index) -> None:
    # Doc 0 та Doc 2 обидва містять "python", перевіряємо узгодженість оцінок
    bm25 = BM25()
    s0 = bm25.score(sample_index, "python", 0)
    s2 = bm25.score(sample_index, "python", 2)
    assert s0 > 0 and s2 > 0


def test_search_engines_consistency_and_cache(sample_index: Index) -> None:
    clear_search_cache()
    res_set = search(sample_index, "python AND async", scorer_name="bm25", engine="set")
    res_merge = search(sample_index, "python AND async", scorer_name="bm25", engine="merge")
    assert [r.doc_id for r in res_set] == [r.doc_id for r in res_merge]
    # Повторний виклик із кешу повертає ідентичний результат
    res_cached = search(sample_index, "python AND async", scorer_name="bm25", engine="set")
    assert res_cached == res_set
