from __future__ import annotations

import os
from typing import Protocol

from .models import SearchHit


class Generator(Protocol):
    def generate(self, query: str, hits: list[SearchHit], observations: list[str]) -> str: ...


class OfflineGenerator:
    """Deterministic generator used for tests, demos, and fair benchmark comparisons."""

    def generate(self, query: str, hits: list[SearchHit], observations: list[str]) -> str:
        if not hits or hits[0].score <= 0:
            return "I do not have enough grounded evidence to answer that request."
        evidence = " ".join(
            f"[{hit.document.id}] {hit.document.text}" for hit in hits if hit.score > 0
        )
        tools = " ".join(observations)
        return f"Grounded answer for: {query}\n\n{evidence}\n\n{tools}".strip()


class OpenAIResponsesGenerator:
    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install the optional OpenAI dependency: pip install -e '.[openai]'"
            ) from exc
        self._client = OpenAI()
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def generate(self, query: str, hits: list[SearchHit], observations: list[str]) -> str:
        context = "\n".join(
            f"[{hit.document.id}] {hit.document.title}: {hit.document.text}" for hit in hits
        )
        instructions = (
            "Answer only from the supplied evidence. Cite every used document "
            "with its bracketed ID. "
            "If evidence is insufficient, say so explicitly."
        )
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=f"Question: {query}\n\nEvidence:\n{context}\n\nTools:\n{observations}",
        )
        return response.output_text


def build_generator(provider: str | None = None) -> Generator:
    provider = (provider or os.getenv("EVIDENCE_RAG_PROVIDER", "offline")).lower()
    if provider == "offline":
        return OfflineGenerator()
    if provider == "openai":
        return OpenAIResponsesGenerator()
    raise ValueError(f"unsupported provider: {provider}")
