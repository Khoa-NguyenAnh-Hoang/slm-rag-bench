---
title: "Results: Blueprint 2"
aliases: ["Results", "Phase C Results"]
type: results
status: draft
blueprint: "[[Blueprint]]"
date: 2026-10-06
tags:
  - results
  - rag/benchmark
related:
  - "[[Blueprint]]"
  - "[[Action Plan]]"
  - "[[Phase 2 Plan]]"
  - "[[Mala]]"
  - "[[EvaRAG]]"
---

# Results — Blueprint 2

> [!warning] Draft skeleton, 2026-10-06
> No real GPU cell has run yet. Values below are placeholders; replace them from committed
> `slm-rag-bench/results/*/scorecard.json` and `slm-rag-bench/results/report.html` after Stage 2.
> The synthetic fixture in `slm-rag-bench/tests/fixture/results/` is for scorecard math only.

## Thesis question

For small-model RAG with a fixed bm25s → MiniLM rerank retriever, does adding CRAG's evaluator
improve the cost–latency–faithfulness frontier, and which generator model should be picked?

## Locked matrix

| Pipeline | Models | Judge | n/cell | Retriever |
|---|---|---:|---:|---|
| `baseline` | 3 | Mistral-7B-Instruct-v0.3 AWQ | 300 | fixed bm25s → MiniLM rerank |
| `crag` | 3 | Mistral-7B-Instruct-v0.3 AWQ | 300 | fixed bm25s → MiniLM rerank |

## Scorecard

Rendered by `python -m analysis.scorecard`. Fill from `results/*/scorecard.json`.

| cell | n | adj acc % | faithfulness | lat p95 ms | ttft p95 ms | API equiv $ | GPU $ | MAP@3 | recall@10 | CRAG action dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline-<model> |  |  |  |  |  |  |  |  |  |  |
| crag-<model> |  |  |  |  |  |  |  |  |  |  |

## Paired crag-vs-baseline tests

Rendered by `python -m analysis.scorecard`; BH-FDR adjusted across comparisons.

| pair | metric | p | q_FDR | interpretation |
|---|---|---:|---:|---|
| crag-<model> vs baseline-<model> | correctness (McNemar) |  |  |  |
| crag-<model> vs baseline-<model> | latency (Wilcoxon) |  |  |  |

## Pareto fronts

Embed or link:

- `results/report.html` — faithfulness vs p95 latency
- `results/report.html` — faithfulness vs total cost

## Human-label agreement

From `python -m src.monitor.labels score` on B2's 50 labels.

| generator | pipeline | agreement % | Cohen's κ | notes |
|---|---|---:|---:|---|
|  |  |  |  |  |

## CRAG validity

From `python tools/crag_validity.py`.

| dataset | action | proxy/reference agreement | takeaway |
|---|---|---:|---|
| popqa | Incorrect | 0% in B4 smoke | flan-t5-base proxy is not discriminative on negatives |

## Failure / deviation log

Carry forward any plan §1 deviations and gate failures here.

| item | observed | consequence |
|---|---|---|
| Self-judge in Phase A | yes | B1 closed with Mistral judge |
| CRAG proxy negatives | non-discriminative | do not interpret Incorrect path as measured accuracy |

## Artifacts

- `slm-rag-bench/results/` — committed scorecards, matrices, per-query JSONL
- `slm-rag-bench/results/report.html` — static report
- `slm-rag-bench/results/scorecard_summary.md` — table + stats
- `slm-rag-bench/Notes/Phase 2 Plan.md` — execution plan
- `slm-rag-bench/README.md` — repro ladder

## Verdict

TBD after Stage 2/3.
