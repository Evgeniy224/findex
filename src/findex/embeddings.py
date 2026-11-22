import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from findex.index import Index
from findex.ranking import SearchResult


class ChunkMeta(TypedDict):
    doc_id: int
    text: str


@lru_cache(maxsize=2)
def _get_model(model_name: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_paths(index_path: Path) -> tuple[Path, Path]:
    return index_path.with_suffix(".embeddings.npy"), index_path.with_suffix(".chunks.json")


def chunk_text(text: str, size: int = 450) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    return [text[start : start + size] for start in range(0, len(text), size)]


def build_embeddings(
    index: Index,
    index_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[int, int]:
    chunks: list[ChunkMeta] = []
    for doc_id, meta in index.doc_meta.items():
        try:
            text = meta.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for chunk in chunk_text(text):
            chunks.append({"doc_id": doc_id, "text": chunk})

    if not chunks:
        raise ValueError("Корпус не містить текстових фрагментів для ембедингів")

    model: Any = _get_model(model_name)
    vectors = cast(
        np.ndarray,
        model.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    vectors_path, chunks_path = embedding_paths(index_path)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(chunks), vectors.shape[1]


def load_embeddings(index_path: Path) -> tuple[np.ndarray, list[ChunkMeta]]:
    vectors_path, chunks_path = embedding_paths(index_path)
    if not vectors_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Ембединг-індекс не знайдено. Спочатку запустіть: findex embed <index>"
        )
    vectors = np.load(vectors_path, mmap_mode="r")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if len(chunks) != vectors.shape[0]:
        raise ValueError("Файли ембедингів пошкоджені: кількість векторів не відповідає чанкам")
    return vectors, chunks


def semantic_search(
    index_path: Path,
    query: str,
    top_k: int,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[SearchResult]:
    vectors, chunks = load_embeddings(index_path)
    model: Any = _get_model(model_name)
    query_vector = np.asarray(
        cast(
            np.ndarray,
            model.encode(query, convert_to_numpy=True, normalize_embeddings=True),
        ),
        dtype=np.float32,
    )
    similarities = vectors @ query_vector

    best_by_doc: dict[int, float] = {}
    for chunk_index, chunk in enumerate(chunks):
        doc_id = int(chunk["doc_id"])
        score = float(similarities[chunk_index])
        best_by_doc[doc_id] = max(best_by_doc.get(doc_id, -np.inf), score)

    ranked = sorted(best_by_doc.items(), key=lambda item: (-item[1], -item[0]))
    return [SearchResult(score=score, doc_id=doc_id) for doc_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    top_k: int,
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (k + rank)
    return [
        SearchResult(score=score, doc_id=doc_id)
        for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], -item[0]))[:top_k]
    ]
