from __future__ import annotations

from pathlib import Path

from .generation import build_generator
from .graph import build_graph
from .models import Citation, QueryResponse
from .retrieval import BM25Retriever, load_documents


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"


class EvidenceRagService:
    def __init__(self, data_path: Path | None = None, provider: str | None = None):
        documents = load_documents(data_path or default_data_path())
        self.retriever = BM25Retriever(documents)
        self.graph = build_graph(self.retriever, build_generator(provider))

    def ask(self, query: str, top_k: int = 3) -> QueryResponse:
        state = self.graph.invoke({"query": query, "top_k": top_k, "trace": []})
        hits = state.get("hits", [])
        return QueryResponse(
            answer=state.get("answer", ""),
            citations=[
                Citation(
                    document_id=hit.document.id,
                    title=hit.document.title,
                    score=round(hit.score, 4),
                )
                for hit in hits
                if hit.score > 0
            ],
            tool_observations=state.get("tool_observations", []),
            trace=state.get("trace", []),
            blocked=state.get("blocked", False),
            block_reason=state.get("block_reason"),
        )
