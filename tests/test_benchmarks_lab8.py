from pathlib import Path
from typing import Any

from findex.index import Index, build_index
from findex.search import clear_search_cache, search


def test_search_benchmark(benchmark: Any, sample_index: Index) -> None:
    clear_search_cache()
    benchmark(lambda: search(sample_index, "python AND async", use_cache=False))


def test_build_index_benchmark(
    benchmark: Any,
    sample_corpus_dir: Path,
) -> None:
    benchmark(build_index, sample_corpus_dir)
