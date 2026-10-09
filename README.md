# slm-rag-bench

A reproducible cost–latency–faithfulness benchmark for small-model RAG.

The benchmark fixes retrieval and varies only the generator pipeline:

```text
HotpotQA distractor
  -> bm25s sparse pool (depth 30)
  -> MiniLM-L6 cross-encoder rerank (top 10)
  -> optional CRAG evaluator: Correct / Incorrect / Ambiguous + context refinement
  -> generator via vLLM / OpenAI-compatible chat endpoint
  -> MetricsLogger (JSONL)
  -> ragas-style faithfulness / answer relevancy / context relevancy
  -> scorecard.json + HTML report
```

## Why the retriever is fixed

The question is **which small model should I pick**, not which retriever works best.
Retrieval therefore stays identical across cells:

- sparse candidate pool: `bm25s`
- reranker: `sentence-transformers/ms-marco-MiniLM-L-6-v2`
- no dense retrieval arm
- no JDK/pyserini dependency

The reranker is part of the fixed retriever, so it runs in every cell and is not an
experimental axis. See `src/retrieval/NOTES.md` for the measurements behind that choice.

## Model / pipeline matrix

The experiment is defined in `configs/experiment.yaml`:

- 3 small generator models
- 2 pipelines: `baseline` and `crag`
- 6 cells total: `(model, pipeline)`
- one fixed judge server for faithfulness/relevance scoring
- fixed corpus and split policy for every cell

`baseline` sends retrieved context directly to the generator.
`crag` first evaluates each retrieved context and can strip it on `Correct` /
`Ambiguous`, or force the generator toward abstention on `Incorrect`.

## Setup

Requires Python 3.11+ and `uv`.

```bash
uv sync --frozen --extra serve
export HF_TOKEN=...    # needed for gated models such as Llama
```

The `serve` extra installs vLLM, OpenAI client, instructor, and ragas.

## Quick check

CPU-only checks (no GPU; the first run downloads HotpotQA and the small encoder models):

```bash
uv run python -m tools.selfcheck
```

Or run the checks individually:

```bash
uv run python -m src.contract
uv run python -m src.monitor.retrieval_metrics
uv run python -m src.monitor.metrics_logger
uv run python -m src.pipeline
uv run python -m src.runner.run_cell --self-check
uv run python -m src.runner.matrix --self-check
```

## Running on a GPU box

Set the two renter-specific values first:

1. `serve.gpu_memory_utilization` in `configs/experiment.yaml`
2. `gpu_usd_per_hour` in `configs/pricing.yaml`

Then:

```bash
python -m tools.selfcheck
python -m src.runner.judge_probe --model-key qwen2.5-1.5b
python -m src.runner.run_cell --only crag-qwen2.5-1.5b --n 5 --limit-corpus 20000
python -m src.runner.run_cell --only crag-qwen2.5-1.5b --n 30
nohup python -m src.runner.matrix --skip-finished > matrix.log 2>&1 &
```

A capped corpus is useful for smoke testing but is **not** a publishable cell; the cap is
recorded in the cell manifest.

`matrix.py` runs each cell in its own subprocess, continues past failures, and writes a
scorecard for successful cells. `--skip-finished` resumes the matrix without re-running
completed cells.

## Outputs

Each finished cell writes:

```text
results/<cell_id>/manifest.json      config, corpus size, pricing version, git rev, hardware
results/<cell_id>/queries.jsonl      one runtime record per query
results/<cell_id>/scorecard.json     aggregate summary + retrieval + adjusted accuracy + ragas
results/scorecard_summary.md         cross-cell table + paired tests
results/report.html                  static scorecard report
```

`queries.jsonl` and `scorecard.json` are committed artifacts. Re-running a cell deletes its
own `queries.jsonl` before starting, so partial output is not mixed with a fresh run.

## Configuration

`configs/experiment.yaml` controls:

- generator models
- baseline / crag pipelines
- judge model and endpoint
- corpus split and evaluation split
- retrieval depth / top-k
- server ports and memory utilisation

`configs/pricing.yaml` controls:

- API-equivalent cost rates
- GPU hourly cost
- provider and verification date for cost assumptions

Costs are reported as two numbers per cell:

- API-equivalent cost, for comparing against hosted-model economics
- GPU-amortised cost, for measuring the actual hardware budget

Neither is allowed to default to `0.0`.

## Measurement contract

The harness is opinionated about what counts as a valid measurement:

1. latency is measured end-to-end, never summed from stages
2. unavailable measurements are `None`, never `0.0`
3. missing reference answers are `Unknown`, never fabricated labels
4. unknown config keys raise instead of falling back silently
5. scores carry their retrieval depth
6. costs carry pricing version and verification metadata
7. the manifest is written before model loading

`src/contract.py` scans the source tree for patterns that violate these rules.

## Repository layout

```text
configs/      experiment and pricing configs
src/          pipeline, retrieval, evaluator, runner, monitor, data loading
analysis/     scorecard and HTML report generation
tools/        selfcheck, labels export/scoring, CRAG validity helper
tests/        CPU smoke tests and synthetic fixtures
results/      committed benchmark artifacts once cells finish
Notes/        project plan and results notes
```

## Limitations

- **Primary metric is faithfulness** (ragas, scored by the AWQ Mistral judge); adjusted
  accuracy, hallucination rate and rejection rate are secondary.
- No cell has produced benchmark numbers yet; the repository currently validates the harness.
- Eval queries are sampled from the same train split that builds the corpus (self-consistent
  by design, see `configs/experiment.yaml`), so results are in-domain: they do not
  generalise out-of-domain, and a small model may answer from parametric memory.
- The 3×2 grid compares pipelines, not isolated CRAG components: the `crag`-vs-`baseline`
  delta bundles the corrective decision, knowledge-strip editing, and evaluator latency.
- Latency is end-to-end as implemented: query 1 includes vLLM server warm-up, `tok_per_s`
  is effective throughput (its denominator includes retrieval/rerank/CRAG), CRAG scoring
  is serial batch-1, and p95 is measured under vLLM's default concurrency (not
  single-stream; `serve.extra_args` can force single-sequence).
- CRAG evaluation is implemented as a local proxy evaluator. Its Incorrect path needs
  external validity checking before it is interpreted as measured accuracy.
- API-equivalent costs depend on published provider prices and can differ from real rented
  GPU cost.
- No dense retriever arm and no live web-search rescue are included.
