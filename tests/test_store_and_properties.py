from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import (
    Index,
    Posting,
    build_index,
    build_index_from_docs,
    intersect_postings,
    union_postings,
)
from findex.store import load_index, open_index, save_index

_WORD_STRATEGY = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=10,
)
_DOC_TEXT_STRATEGY = st.lists(_WORD_STRATEGY, min_size=0, max_size=25).map(" ".join)
_CORPUS_STRATEGY = st.lists(_DOC_TEXT_STRATEGY, min_size=1, max_size=15)


def test_save_and_open_index_roundtrip(sample_index: Index, tmp_path: Path) -> None:
    out_file = tmp_path / "saved_index.json"
    save_index(sample_index, out_file)
    with open_index(out_file) as loaded:
        assert loaded == sample_index


@given(texts=_CORPUS_STRATEGY)
def test_property_postings_always_sorted(texts: list[str]) -> None:
    """Властивість 1 (Hypothesis): постінги для кожного терму строго відсортовані за doc_id."""
    docs = [Document(i, Path(f"doc_{i}.txt"), text) for i, text in enumerate(texts)]
    index = build_index_from_docs(docs)
    for plist in index.postings.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids)
        assert len(doc_ids) == len(set(doc_ids))


@given(texts=_CORPUS_STRATEGY)
def test_property_save_load_roundtrip(
    texts: list[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Властивість 2 (Hypothesis): load(save(index)) завжди дає ідентичний індекс."""
    docs = [Document(i, Path(f"doc_{i}.txt"), text) for i, text in enumerate(texts)]
    index = build_index_from_docs(docs)
    tmp_dir = tmp_path_factory.mktemp("hyp_store")
    file_path = tmp_dir / "idx.json"
    save_index(index, file_path)
    restored = load_index(file_path)
    assert restored == index


@given(
    left_ids=st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted),
    right_ids=st.lists(st.integers(min_value=0, max_value=200), unique=True).map(sorted),
)
def test_property_merge_matches_set_operations(left_ids: list[int], right_ids: list[int]) -> None:
    """Властивість 3 (Hypothesis): two-pointer merge збігається з операціями над set."""
    left_postings = [Posting(doc_id=d, tf=1) for d in left_ids]
    right_postings = [Posting(doc_id=d, tf=1) for d in right_ids]

    assert intersect_postings(left_postings, right_postings) == sorted(
        set(left_ids) & set(right_ids)
    )
    assert union_postings(left_postings, right_postings) == sorted(
        set(left_ids) | set(right_ids)
    )


@pytest.mark.slow
def test_large_synthetic_corpus_indexing(tmp_path: Path) -> None:
    """Повільний інтеграційний тест на 1000 документах."""
    for i in range(1000):
        (tmp_path / f"doc_{i:04d}.txt").write_text(
            f"Документ номер {i} пошукова система python async індекс",
            encoding="utf-8",
        )
    idx = build_index(tmp_path)
    assert idx.num_docs == 1000
    assert idx.df("пошукова") == 1000