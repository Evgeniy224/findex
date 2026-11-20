from pathlib import Path

from findex.index import (
    Index,
    Posting,
    build_index,
    intersect_postings,
    union_postings,
)


def test_build_index_metadata_and_terms(sample_index: Index) -> None:
    assert sample_index.num_docs == 4
    assert "python" in sample_index
    assert "неіснуюче" not in sample_index
    assert sample_index.df("python") == 2
    assert sample_index.df("пошукова") == 2
    assert sample_index.avg_doc_length > 0


def test_index_mapping_protocol(sample_index: Index) -> None:
    terms = list(sample_index)
    assert len(terms) == len(sample_index)
    assert repr(sample_index).startswith("Index(terms=")
    assert sample_index.get_doc_meta(999) is None


def test_empty_corpus_index(tmp_path: Path) -> None:
    empty_index = build_index(tmp_path)
    assert empty_index.num_docs == 0
    assert empty_index.avg_doc_length == 0.0
    assert len(empty_index) == 0


def test_intersect_postings_two_pointers() -> None:
    left = [Posting(0, 1), Posting(2, 3), Posting(5, 1), Posting(9, 2)]
    right = [Posting(1, 1), Posting(2, 1), Posting(4, 2), Posting(9, 1)]
    assert intersect_postings(left, right) == [2, 9]


def test_union_postings_two_pointers() -> None:
    left = [Posting(0, 1), Posting(3, 2)]
    right = [Posting(2, 1), Posting(3, 1), Posting(7, 1)]
    assert union_postings(left, right) == [0, 2, 3, 7]