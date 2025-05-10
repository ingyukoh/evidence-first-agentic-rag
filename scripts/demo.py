from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.service import EvidenceRagService  # noqa: E402


def main() -> None:
    service = EvidenceRagService(provider="offline")
    query = "What controls make an agent tool workflow safe, and how can I prove the evidence?"
    response = service.ask(query)
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
