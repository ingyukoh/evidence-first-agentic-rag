from fastapi import FastAPI

from .models import QueryRequest, QueryResponse
from .service import EvidenceRagService

app = FastAPI(
    title="Evidence-First Agentic RAG",
    version="0.1.0",
    description="Agentic retrieval with transparent evaluation, safety checks, and traces.",
)
service = EvidenceRagService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "configured"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return service.ask(request.query, request.top_k)
