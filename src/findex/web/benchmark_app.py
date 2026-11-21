import logging
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from findex.index import Index
from findex.query_parser import QuerySyntaxError
from findex.search import search
from findex.store import load_index
from findex.web.app import get_index
from findex.web.models import SearchResponse, SearchResultModel
from findex.web.settings import get_settings

logger = logging.getLogger(__name__)
_mode = os.environ.get("BENCHMARK_MODE", "def")


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    application.state.index = load_index(settings.index_path)
    yield
    application.state.index = None


app = FastAPI(title="findex load test", lifespan=lifespan)


def _search_response(
    index: Index | None,
    q: str,
    k: int,
    scorer: Literal["bm25", "tfidf"],
) -> SearchResponse:
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    started = time.perf_counter()
    try:
        all_results = search(
            index,
            query=q,
            scorer_name=scorer,
            top_k=index.num_docs,
            use_cache=False,
        )
    except QuerySyntaxError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total = len(all_results)
    return SearchResponse(
        query=q,
        total=total,
        took_ms=(time.perf_counter() - started) * 1000,
        results=[
            SearchResultModel(
                doc_id=result.doc_id,
                title=(
                    meta.path.name
                    if (meta := index.get_doc_meta(result.doc_id)) is not None
                    else f"doc_{result.doc_id}"
                ),
                score=result.score,
                snippet=result.snippet,
            )
            for result in all_results[:k]
        ],
        page=1,
        pages=math.ceil(total / k),
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "index", None) is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return {"status": "ok"}


if _mode == "def":

    @app.get("/search", response_model=SearchResponse)
    def search_sync(
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
    ) -> SearchResponse:
        return _search_response(index, q, k, scorer)

elif _mode == "async":

    @app.get("/search", response_model=SearchResponse)
    async def search_async(
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
    ) -> SearchResponse:
        return _search_response(index, q, k, scorer)

else:
    raise ValueError(f"Unsupported BENCHMARK_MODE: {_mode}")
