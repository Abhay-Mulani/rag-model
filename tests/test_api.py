import os

os.environ["LLM_PROVIDER"] = "local"
os.environ["ENABLE_SEMANTIC_RETRIEVAL"] = "false"

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_grounded_response() -> None:
    response = client.post(
        "/query",
        json={"query": "What are the risks of a concentrated portfolio?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert "finance-research.md" in payload["sources"][0]
    assert "validator" in payload["agents_invoked"]
    assert payload["intent"] == "risk"


def test_query_validates_short_input() -> None:
    response = client.post("/query", json={"query": "hi"})
    assert response.status_code == 422
