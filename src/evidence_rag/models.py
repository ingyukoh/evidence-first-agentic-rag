from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


class TraceEvent(BaseModel):
    node: str
    duration_ms: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=3, ge=1, le=8)


class Citation(BaseModel):
    document_id: str
    title: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    tool_observations: list[str]
    trace: list[TraceEvent]
    blocked: bool = False
    block_reason: str | None = None


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    blocked: bool
    block_reason: str | None
    hits: list[SearchHit]
    tool_observations: list[str]
    answer: str
    trace: list[TraceEvent]
