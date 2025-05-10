from fastapi.testclient import TestClient

from evidence_rag.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_contract() -> None:
    response = client.post("/query", json={"query": "How should RAG grounding be evaluated?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["blocked"] is False
