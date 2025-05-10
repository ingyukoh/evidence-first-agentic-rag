from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.service import EvidenceRagService  # noqa: E402


def load_cases() -> list[dict]:
    with (ROOT / "evals" / "dataset.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(top_k: int) -> dict:
    service = EvidenceRagService(provider="offline")
    rows: list[dict] = []
    for case in load_cases():
        started = perf_counter()
        response = service.ask(case["query"], top_k=top_k)
        elapsed_ms = (perf_counter() - started) * 1_000
        returned = {citation.document_id for citation in response.citations}
        expected = set(case["expected_docs"])
        recall = 1.0 if not expected else len(returned & expected) / len(expected)
        required = [term.lower() for term in case["required_terms"]]
        term_coverage = (
            1.0
            if not required
            else sum(term in response.answer.lower() for term in required) / len(required)
        )
        block_correct = response.blocked == case["blocked"]
        task_success = recall == 1.0 and term_coverage == 1.0 and block_correct
        rows.append(
            {
                "id": case["id"],
                "recall": recall,
                "term_coverage": term_coverage,
                "block_correct": block_correct,
                "task_success": task_success,
                "latency_ms": elapsed_ms,
            }
        )
    return {
        "top_k": top_k,
        "retrieval_recall": statistics.mean(row["recall"] for row in rows),
        "term_coverage": statistics.mean(row["term_coverage"] for row in rows),
        "task_success": statistics.mean(row["task_success"] for row in rows),
        "p50_latency_ms": statistics.median(row["latency_ms"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    results = {"baseline_top1": evaluate(1), "agentic_top3": evaluate(3)}
    output_json = ROOT / "evals" / "results.json"
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Evaluation results",
        "",
        "Generated from the committed deterministic dataset. Lower latency is better; "
        "other metrics are higher-is-better.",
        "",
        "| System | Retrieval recall | Required-term coverage | Task success | p50 latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['retrieval_recall']:.1%} | {result['term_coverage']:.1%} | "
            f"{result['task_success']:.1%} | {result['p50_latency_ms']:.2f} |"
        )
    lines.extend(["", "Run `make benchmark` to reproduce these numbers."])
    (ROOT / "evals" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
