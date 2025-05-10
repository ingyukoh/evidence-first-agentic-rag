from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Document, SearchHit

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.-]*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            documents.append(
                Document(
                    id=raw["id"],
                    title=raw["title"],
                    text=raw["text"],
                    tags=tuple(raw.get("tags", [])),
                )
            )
    return documents


class BM25Retriever:
    """Small transparent BM25 implementation for reproducible offline evaluation."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("at least one document is required")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = [
            tokenize(f"{doc.title} {doc.text} {' '.join(doc.tags)}") for doc in documents
        ]
        self._counts = [Counter(tokens) for tokens in self._tokens]
        self._avg_len = sum(map(len, self._tokens)) / len(self._tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        total = len(documents)
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        query_terms = tokenize(query)
        scored: list[SearchHit] = []
        for document, counts, tokens in zip(
            self.documents, self._counts, self._tokens, strict=True
        ):
            score = 0.0
            length_normalizer = 1 - self.b + self.b * len(tokens) / self._avg_len
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_normalizer
                score += self._idf.get(term, 0.0) * numerator / denominator
            scored.append(SearchHit(document=document, score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.document.id))
        return scored[:top_k]
