# Evaluation results

Generated from the committed deterministic dataset. Lower latency is better; other metrics are higher-is-better.

| System | Retrieval recall | Required-term coverage | Task success | p50 latency (ms) |
|---|---:|---:|---:|---:|
| baseline_top1 | 93.8% | 100.0% | 87.5% | 0.45 |
| agentic_top3 | 100.0% | 100.0% | 100.0% | 0.47 |

Run `make benchmark` to reproduce these numbers.
