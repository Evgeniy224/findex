from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web import create_app
from findex.web.app import get_index


@pytest.fixture
def client(sample_index: Index) -> Iterator[TestClient]:
    application = create_app(index_loader=lambda _path: sample_index)
    application.dependency_overrides[get_index] = lambda: sample_index
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def test_search_returns_paginated_results_and_request_id(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python", "k": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "python"
    assert payload["total"] == 2
    assert len(payload["results"]) == 1
    assert payload["pages"] == 2
    assert response.headers["x-request-id"]
    second_page = client.get("/search", params={"q": "python", "k": 1, "page": 2})
    assert second_page.json()["results"][0]["doc_id"] != payload["results"][0]["doc_id"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "python", "k": 0},
        {"q": "python", "k": 101},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "unknown"},
    ],
)
def test_search_validates_parameters(client: TestClient, params: dict[str, str | int]) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_search_rejects_invalid_query_syntax(client: TestClient) -> None:
    response = client.get("/search", params={"q": "(python AND"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_document_endpoint_returns_text_and_404_for_missing_id(client: TestClient) -> None:
    response = client.get("/docs/0")
    assert response.status_code == 200
    assert "Python async" in response.json()["text"]
    assert client.get("/docs/999").status_code == 404


def test_stats_and_health(client: TestClient) -> None:
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["doc_count"] == 4
    assert stats.json()["vocabulary"] > 0
    assert client.get("/health").json() == {"status": "ok"}


def test_health_returns_503_when_index_is_unavailable(sample_index: Index) -> None:
    application = create_app(index_loader=lambda _path: sample_index)
    application.dependency_overrides[get_index] = lambda: None
    with TestClient(application) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_search_page_and_document_view(client: TestClient) -> None:
    page = client.get("/", params={"q": "python"})
    assert page.status_code == 200
    assert "findex" in page.text
    assert "<mark>" in page.text
    assert client.get("/doc/0").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/docs").status_code == 200
    assert "/search" in client.get("/openapi.json").text


def test_unexpected_error_is_clean_and_has_request_id(sample_index: Index) -> None:
    application = create_app(index_loader=lambda _path: sample_index)

    @application.get("/failure")
    def failure() -> None:
        raise RuntimeError("sensitive exception detail")

    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.get("/failure")
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "sensitive exception detail" not in response.text
    assert response.headers["x-request-id"] == response.json()["request_id"]


def test_lifespan_loads_index_once(sample_index: Index, tmp_path: Path) -> None:
    calls = 0

    def loader(_path: Path) -> Index:
        nonlocal calls
        calls += 1
        return sample_index

    application = create_app(index_loader=loader)
    with TestClient(application) as test_client:
        test_client.get("/health")
        test_client.get("/search", params={"q": "python"})
    assert calls == 1
