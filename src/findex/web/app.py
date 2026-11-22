import logging
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import RequestResponseEndpoint

from findex.index import Index
from findex.query_parser import QuerySyntaxError
from findex.search import search
from findex.store import load_index
from findex.web.models import (
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    SearchResultModel,
    StatsResponse,
)
from findex.web.settings import get_settings

logger = logging.getLogger(__name__)
_PACKAGE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))


def get_index(request: Request) -> Index | None:
    return getattr(request.app.state, "index", None)


def _read_document(index: Index, doc_id: int) -> DocumentResponse:
    meta = index.get_doc_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        text = meta.path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.info("Document %s cannot be read from %s: %s", doc_id, meta.path, exc)
        raise HTTPException(status_code=404, detail="Document content is unavailable") from exc
    return DocumentResponse(
        doc_id=doc_id,
        title=meta.path.name or str(meta.path),
        path=str(meta.path),
        text=text,
    )


def _highlight_parts(snippet: str) -> list[tuple[str, bool]]:
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for match in re.finditer(r"\*([^*]+)\*", snippet):
        if match.start() > cursor:
            parts.append((snippet[cursor : match.start()], False))
        parts.append((match.group(1), True))
        cursor = match.end()
    if cursor < len(snippet):
        parts.append((snippet[cursor:], False))
    return parts or [(snippet, False)]


def create_app(index_loader: Callable[[Path], Index] = load_index) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        settings = get_settings()
        application.state.index = None
        application.state.started_at = time.perf_counter()
        try:
            application.state.index = index_loader(settings.index_path)
            logger.info("Search index loaded from %s", settings.index_path)
            yield
        except Exception:
            logger.exception("Could not load or serve the search index")
            raise
        finally:
            application.state.index = None

    application = FastAPI(
        title="findex API",
        description="HTTP API and web interface for the findex search engine.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(_PACKAGE_DIR / "static")),
        name="static",
    )

    @application.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid4()))
        logger.exception("Unhandled request error (request_id=%s)", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    @application.get("/search", response_model=SearchResponse)
    def search_route(
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
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
                mode=mode,
                index_path=get_settings().index_path,
            )
        except QuerySyntaxError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        total = len(all_results)
        pages = (total + k - 1) // k
        start = (page - 1) * k
        results = [
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
            for result in all_results[start : start + k]
        ]
        return SearchResponse(
            query=q,
            total=total,
            took_ms=(time.perf_counter() - started) * 1000,
            results=results,
            page=page,
            pages=pages,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def get_document(
        doc_id: int,
        index: Annotated[Index | None, Depends(get_index)],
    ) -> DocumentResponse:
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not loaded")
        return _read_document(index, doc_id)

    @application.get("/stats", response_model=StatsResponse)
    def stats_route(
        request: Request,
        index: Annotated[Index | None, Depends(get_index)],
    ) -> StatsResponse:
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not loaded")
        started_at = getattr(request.app.state, "started_at", time.perf_counter())
        return StatsResponse(
            index_size=sum(len(postings) for postings in index.postings.values()),
            doc_count=index.num_docs,
            vocabulary=len(index),
            uptime_seconds=max(0.0, time.perf_counter() - started_at),
        )

    @application.get("/health", response_model=HealthResponse)
    def health_route(
        index: Annotated[Index | None, Depends(get_index)],
    ) -> HealthResponse | JSONResponse:
        if index is None:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return HealthResponse(status="ok")

    @application.get("/", include_in_schema=False)
    def home(
        request: Request,
        index: Annotated[Index | None, Depends(get_index)],
        q: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        page: Annotated[int, Query(ge=1)] = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: Literal["keyword", "semantic", "hybrid"] = "keyword",
    ):
        results: list[dict[str, object]] = []
        total = 0
        pages = 0
        error = None
        if q and index is not None:
            try:
                all_results = search(
                    index,
                    q,
                    scorer_name=scorer,
                    top_k=index.num_docs,
                    mode=mode,
                    index_path=get_settings().index_path,
                )
                total = len(all_results)
                pages = (total + k - 1) // k
                start = (page - 1) * k
                for result in all_results[start : start + k]:
                    meta = index.get_doc_meta(result.doc_id)
                    results.append(
                        {
                            "doc_id": result.doc_id,
                            "title": meta.path.name if meta is not None else f"doc_{result.doc_id}",
                            "score": result.score,
                            "parts": _highlight_parts(result.snippet),
                        }
                    )
            except QuerySyntaxError as exc:
                error = str(exc)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "q": q or "",
                "k": k,
                "page": page,
                "pages": pages,
                "total": total,
                "scorer": scorer,
                "mode": mode,
                "results": results,
                "error": error,
            },
        )

    @application.get("/doc/{doc_id}", include_in_schema=False)
    def document_page(
        request: Request,
        doc_id: int,
        index: Annotated[Index | None, Depends(get_index)],
    ):
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not loaded")
        document = _read_document(index, doc_id)
        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={"document": document},
        )

    return application


app = create_app()
