# slm-rag-bench

**Blueprint 2 — A Cost-Latency-Faithfulness Benchmark for Small-Model RAG** (GAP 5).
Design + phase plan: `../Notes/Phase 2 Plan.md`.

```
HotpotQA distractor → HybridLI (BM25 ×0.7 + MPNet/FAISS ×0.3) → [MiniLM-L6 rerank ×0.85]
→ [CRAG evaluator: Correct / Incorrect / Ambiguous + strip refinement]
→ generator via vLLM (OpenAI chat, streaming) → MetricsLogger (JSONL)
→ ragas 0.4.3 collections (Faithfulness + AnswerRelevancy + ContextRelevance) → scorecard
```

The brackets are the ablation: `baseline` / `reranked` / `crag` are three `Pipeline` specs over
one timed `answer_query()` (`src/pipeline.py`), not three code paths. Grid = 3 models × 3
pipelines = 9 cells (`configs/experiment.yaml`).

## Setup

```bash
uv sync --extra sparse                     # + rank-bm25: no-JDK sparse arm, for local iteration
uv sync --extra retrieval --extra serve    # Colab T4: pyserini + vLLM + instructor + ragas
```

- `pyserini` (retrieval) needs **JDK 17+**:
  `apt-get install -y openjdk-17-jdk-headless`. It is the default sparse arm because it matches
  the reference BM25 latency (~6.7 ms/query on the 90k index, Mala §3); `rank_bm25` measures
  ~117 ms at 50k docs and would dominate end-to-end latency at ~904k. Set `retrieval.sparse:
  rank_bm25` to measure the gap on your own hardware — it is a documented deviation either way.
- `ragas>=0.4.3` from **PyPI** (collections API). `LangchainLLMWrapper` is a hard error there, not
  legacy — the judge must be an instructor-style client over the vLLM endpoint
  (`src/runner/serve.py`). Verified against the 0.4.3 wheel: `llm_factory(client=)`,
  `metrics.collections.{Faithfulness,AnswerRelevancy,ContextRelevance}`,
  `embeddings.HuggingFaceEmbeddings`. `serve.ragas_compat()` stubs the module ragas cannot
  import without — see its docstring for why a `sys.modules` stub is required in-process.

## Run order

```bash
uv run python -m tools.selfcheck          # ALL CPU checks, one command, no GPU/network needed

# or individually:
uv run python -m src.monitor.retrieval_metrics
uv run python -m src.monitor.metrics_logger
uv run python -m src.pipeline
uv run python -m src.runner.run_cell --self-check
uv run python -m src.data.hotpot_loader   # downloads HotpotQA once, caches to scratch/
uv run python -m src.retrieval.hybrid_reranker   # rank_bm25 arm, CPU
uv run python -m src.evaluator.crag_module

# GPU + vLLM. Probe FIRST: ~2 min to validate streaming/usage/TTFT and the judge,
# versus discovering a broken judge after a 30-minute generation pass.
uv run python -m src.runner.judge_probe --model-key qwen3-4b
uv run python -m src.runner.run_cell --only crag-qwen3-4b --n 30      # pilot
uv run python -m src.runner.run_cell --only crag-qwen3-4b            # full cell, n=300
```

Add `--limit-corpus 20000` for a fast local run. A capped corpus inflates every retrieval
metric, so it is recorded in the manifest — a capped run is **not** a publishable cell.

### On Colab (T4)

There is no notebook generator yet (`tools/build_notebook.py` is Phase C, C4). Today the pilot
runs directly in a Colab cell — no `%%writefile` indirection, so the notebook can never drift
from `src/`:

```bash
!apt-get install -y -qq openjdk-17-jdk-headless          # pyserini needs JDK 17
!pip install -q vllm "ragas>=0.4.3" instructor openai datasets rank-bm25 \
                 sentence-transformers faiss-cpu pyyaml pandas
!huggingface-cli login                                  # Llama 3.1 is gated
!git clone https://github.com/<you>/slm-rag-bench.git && cd slm-rag-bench
!python -m tools.selfcheck && python -m src.runner.judge_probe --model-key qwen3-4b
!python -m src.runner.run_cell --only crag-qwen3-4b --n 30
```

**The index cache does not survive a Colab restart.** `scratch/index/<hash>/` costs 30–60 min to
rebuild at full corpus scale and is gitignored, so attach a Drive mount or keep the session
alive across cells of the same run.

## Two things that will silently produce garbage

- **`eval_split` must equal `corpus.split`** (`configs/experiment.yaml`). A dev query's gold
  paragraphs are not in the train index, so scoring dev against train measures corpus coverage,
  not retrieval quality. `run_cell` aborts before loading a model if any query's gold is absent.
- **TTFT needs `stream=True`.** `serve_generate` raises if the server returns no usage block —
  token counts, and therefore the entire cost axis, depend on it. Never paper over that with a 0.

## Outputs

```
results/<cell_id>/manifest.json    config + corpus size + pricing card + git rev + hardware
                       /queries.jsonl  one record per query (COMMITTED — not gitignored)
                       /scorecard.json  summary, retrieval metrics, adjusted accuracy, ragas
scratch/index/<hash>/  BM25 + embedding caches. The 90,447-row corpus build is 30–60 min,
                      built once and reused by all 9 cells. Do not delete between cells.
```

Retrieval metrics are **depth-parameterised** (Mala's `num_relevant = sum(rel)` grows with
retrieval depth), so every number carries its own `depth` and cannot be compared across runs at
different depths. See the `src/monitor/retrieval_metrics.py` docstring before quoting a MAP.

## Current status

Phase A complete: observed end-to-end latency with co-resident retrieval models, split
retrieve/rerank stages, real corpus over the HF route (the raw-JSON hosts are dead), retrieval
quality metrics, and two labelled cost numbers. Phase B (separate judge endpoint, LLM-judge
labels + hand-annotated agreement sample, CRAG proxy validation) not started.
