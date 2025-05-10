from __future__ import annotations

from collections import Counter

from .models import SearchHit
from .retrieval import tokenize


def analyze_skill_gap(query: str, hits: list[SearchHit]) -> str:
    requested = set(tokenize(query))
    evidence = Counter(
        token for hit in hits for token in (*hit.document.tags, *tokenize(hit.document.title))
    )
    matched = sorted(term for term in requested if term in evidence)
    unmatched = sorted(term for term in requested if term not in evidence and len(term) > 3)
    return (
        f"Skill-gap tool: supported terms={matched[:8] or ['none']}; "
        f"terms requiring additional evidence={unmatched[:8] or ['none']}."
    )


def inspect_evidence_quality(hits: list[SearchHit]) -> str:
    direct = [hit.document.id for hit in hits if "direct-evidence" in hit.document.tags]
    transferable = [hit.document.id for hit in hits if "transferable" in hit.document.tags]
    return (
        "Evidence-quality tool: "
        f"direct={direct or ['none']}; transferable={transferable or ['none']}."
    )


def select_tools(query: str) -> list[str]:
    lowered = query.lower()
    tools: list[str] = []
    if any(term in lowered for term in ("skill", "gap", "requirement", "match")):
        tools.append("skill_gap")
    if any(term in lowered for term in ("evidence", "prove", "portfolio", "project")):
        tools.append("evidence_quality")
    return tools
