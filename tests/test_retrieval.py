from pathlib import Path

from evidence_rag.retrieval import BM25Retriever, load_documents

DATA = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"


def test_retrieves_grounding_document() -> None:
    retriever = BM25Retriever(load_documents(DATA))
    hits = retriever.search("grounded retrieval citations recall", top_k=3)
    assert hits[0].document.id == "rag-grounding"
    assert hits[0].score > 0


def test_results_are_deterministic() -> None:
    retriever = BM25Retriever(load_documents(DATA))
    first = [hit.document.id for hit in retriever.search("agent tools approval", top_k=4)]
    second = [hit.document.id for hit in retriever.search("agent tools approval", top_k=4)]
    assert first == second
