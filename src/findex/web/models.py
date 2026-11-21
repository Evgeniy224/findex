from pydantic import BaseModel, ConfigDict


class SearchResultModel(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    results: list[SearchResultModel]
    page: int
    pages: int


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsResponse(BaseModel):
    index_size: int
    doc_count: int
    vocabulary: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
