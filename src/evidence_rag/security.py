from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all |any )?(previous|prior) instructions",
        r"reveal (the )?(system|developer) prompt",
        r"exfiltrat(e|ion)",
        r"print (all )?(secrets|credentials|environment variables)",
        r"bypass (the )?(guardrail|safety|policy)",
    )
]


def detect_prompt_injection(text: str) -> str | None:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return "The request contains a prompt-injection pattern and was not executed."
    return None


def has_citations(answer: str, document_ids: list[str]) -> bool:
    return bool(document_ids) and all(f"[{document_id}]" in answer for document_id in document_ids)
