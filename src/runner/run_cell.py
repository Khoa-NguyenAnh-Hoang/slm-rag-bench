"""Run ONE cell of the grid: <model> x <pipeline> over n queries, end-to-end, on the GPU.

Four decisions that are not obvious from the code:

  * NO STAGE BATCHING. Retrieval and CRAG run per query inside the same timed chain, with the
    retrieval models co-resident with the vLLM server (MPNet+MiniLM+flan-t5 ~1.5 GB + vLLM 8 GB
    < 16 GB T4). Batching the stages and reporting sum(stage_latency_ms) yields a sum of
    separately measured distributions, not an observed end-to-end number — and latency is the
    whole point. One outer timer spans retrieve -> rerank -> crag -> generate per query.
  * The pipeline is a spec, not a code path (src/pipeline.py).
  * Cost is real: api_cost_usd + gpu_cost_usd from configs/pricing.yaml.
  * TTFT requires stream=True; a missing usage block RAISES rather than returning zeros.
  * manifest.json is written BEFORE any model loads, so a run that dies partway is still
    attributable (on Colab, disconnection is a matter of when, not if).

Run (from the repo root, needs a GPU + vLLM):
    uv run python -m src.runner.run_cell --only crag-qwen3-4b
    uv run python -m src.runner.run_cell --only baseline-qwen2.5-1.5b --n 30   # pilot
Probe first (fails in ~2 min instead of 30):
    uv run python -m src.runner.judge_probe --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

from src.data.hotpot_loader import build_corpus, dataset_manifest, sample_hotpot
from src.monitor.metrics_logger import MetricsLogger, QueryRecord, load_pricing
from src.monitor.retrieval_metrics import aggregate, score_run
from src.pipeline import PIPELINES, adjusted_accuracy, answer_query, label

MODEL_KEYS = ("qwen2.5-1.5b", "qwen3-4b", "llama3.1-8b")


def cell_id(pipeline: str, model_key: str) -> str:
    return f"{pipeline}-{model_key}"


def resolve(cfg: dict, only: str) -> tuple[str, str]:
    """'--only crag-qwen3-4b' -> (pipeline_name, model_key). Fails loudly on an unknown cell
    rather than defaulting, so a typo cannot silently run the wrong experiment."""
    if not only:
        raise SystemExit("--only <pipeline>-<model_key> is required, e.g. crag-qwen3-4b")
    candidates = [p for p in PIPELINES if only.startswith(p)]
    if not candidates:
        raise SystemExit(f"unknown pipeline in {only!r}; have {sorted(PIPELINES)}")
    pipeline = max(candidates, key=len)
    model_key = only[len(pipeline) + 1:]
    if model_key not in MODEL_KEYS:
        raise SystemExit(f"unknown model {model_key!r} in {only!r}; have {list(MODEL_KEYS)}")
    return pipeline, model_key


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def write_manifest(out_dir: Path, cfg: dict, pipeline: str, model_key: str,
                   n: int, corpus_info: dict, pricing: dict) -> None:
    """Written FIRST, before any model load. Records everything needed to re-run this cell."""
    import torch

    cap = corpus_info.get("max_paras")      # a capped corpus is not a publishable cell

    m = cfg["models"][model_key]
    (out_dir / "manifest.json").write_text(json.dumps({
        "cell": cell_id(pipeline, model_key),
        "pipeline": pipeline,
        "model_key": model_key,
        "model_id": m["model_id"],
        "serve_id": m["serve_id"],
        "quantization": m.get("quantization"),
        "n": n,
        "seed": cfg["seed"],
        "config": cfg,
        "dataset": dataset_manifest(),
        "corpus": corpus_info,
        "pricing": pricing,
        "git_rev": git_rev(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        },
        "deviations": [
            "judge == generator: judge.serve_id is null, so both roles resolve to one endpoint",
            "CRAG evaluator is the flan-t5-base proxy, not CRAG's trained 0.77B t5-large",
            f"sparse arm = {cfg['retrieval']['sparse']}",
            f"api prices unverified: pricing version {pricing['version']}, "
            f"verified={pricing['verified']} (see configs/pricing.yaml)",
        ] + ([f"CORPUS CAPPED at {cap} paragraphs — every retrieval metric is inflated and this "
              "cell is NOT publishable"] if cap else []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def ragas_scores(samples: list[dict], judge, embeddings) -> dict[str, float]:
    """Reference-free Faithfulness + AnswerRelevancy via the ragas collections API.

    The `ragas.metrics` singletons are importable but emit DeprecationWarning and route to the
    same validation that rejects local judges (metrics/collections/base.py:113) — the collections
    import is the honest path.

    Returns None-valued metrics as absent rather than zero, so a scoring failure is visible
    instead of being reported as a 0.0 faithfulness.
    """
    from src.runner.serve import ragas_compat

    ragas_compat()          # must precede the ragas import
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness

    ds = EvaluationDataset.from_list([
        {"user_input": s["question"],
         "retrieved_contexts": s["contexts"] or [""],   # "" == the abstention branch
         "response": s["answer"]}
        for s in samples
    ])
    df = evaluate(ds, metrics=[Faithfulness(llm=judge),
                               AnswerRelevancy(llm=judge, embeddings=embeddings),
                               ContextRelevance(llm=judge)],
                  llm=judge, embeddings=embeddings).to_pandas()
    out: dict[str, float] = {}
    for col in ("faithfulness", "answer_relevancy", "context_relevance"):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals):
            out[col] = float(vals.mean())
        out[f"{col}_n"] = int(len(vals))
    return out


def phase_a_gates(pipeline, results: list[dict], summary: dict, scorecard: dict,
                  n: int, records=(), runs=()) -> dict[str, bool]:
    """The acceptance checks a finished cell must pass, as a pure function.

    Pure so it can be tested without a GPU: a gate that only ever runs after a 30-minute Colab
    pass is a gate that gets trusted. `python -m src.runner.run_cell` self-checks this below.

    The last four gates are `src/contract.py` — the measurement discipline is enforced as gates,
    not as prose, so a fabricated zero or an invented verdict fails the run instead of shipping.
    """
    from src.contract import UNKNOWN, check_record, check_scores, scan_source

    violations = [v for rec in records for v in check_record(rec)]
    return {
        "all answers non-empty": all(r["answer"].strip() for r in results),
        "queries.jsonl complete": summary["n_queries"] == n,
        # Regression guard for contract rule 1: an unobserved latency (a sum of stage
        # distributions) is the exact failure this benchmark exists to avoid, so it must fail the
        # run, not annotate it.
        "observed latency dominates stages": all(
            r["latency_ms"] >= sum(r["stage_latency_ms"].values()) - 1e-6 for r in results),
        "P95 latency present": (summary["latency_ms"]["p95"] or 0) > 0,
        "per-stage P95 present": {"retrieve", "generate"} <= set(summary["stage_latency_ms"]),
        "rerank is its own stage (reranked/crag only)": (
            "rerank" in summary["stage_latency_ms"] if pipeline.rerank else True),
        "crag is its own stage (crag only)": (
            "crag" in summary["stage_latency_ms"] if pipeline.correct else True),
        "TTFT measured on every query": all((r["ttft_ms"] or 0) > 0 for r in results),
        "retrieval depth recorded": bool(results) and bool(results[0]["retrieved_ids"]),
        "cost is not zero": (summary["api_cost_usd"] or 0) > 0,
        "adjusted accuracy reported": scorecard["adjusted_accuracy_pct"] is not None,
        "ragas faithfulness scored": scorecard["ragas"].get("faithfulness_n", 0) == n,
        "CRAG has a non-Correct trigger": (
            any(k != "Correct" for k in scorecard["crag_action_distribution"]) if pipeline.correct
            else True),
        # --- contract (src/contract.py) ---
        "contract: no fabricated measurement": not violations,
        "contract: no fabricated verdict": all(
            r["extra"].get("label") != UNKNOWN for r in results),
        "contract: retrieval scores carry metadata": not [
            v for r, s in zip(results, runs) for v in check_scores(s, r["gold_ids"])],
        "contract: source has no rule-2 violation": not scan_source("src"),
    }


def main() -> int:
    from src.contract import UNKNOWN

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--only", help="cell id, e.g. crag-qwen3-4b")
    ap.add_argument("--n", type=int, help="override the eval size (pilot runs)")
    ap.add_argument("--limit-corpus", type=int, help="cap corpus paragraphs (local iteration "
                                                     "ONLY — inflates every retrieval metric)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pipeline_name, model_key = resolve(cfg, args.only)
    # top_k comes from the config, not from the Pipeline default. Leaving it to the dataclass
    # would give `retrieval.top_k` two sources of truth and editing the YAML would silently do
    # nothing — the config would describe a run that never happened.
    pipeline = replace(PIPELINES[pipeline_name], top_k=int(cfg["retrieval"]["top_k"]))
    n = args.n or cfg["n"]
    cfg["_model_key"] = model_key

    cid = cell_id(pipeline_name, model_key)
    out_dir = Path(cfg["output_root"]) / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queries.jsonl").unlink(missing_ok=True)   # logger appends; a stale file breaks gates

    # ---- data ----------------------------------------------------------------
    rows = sample_hotpot(n=n, seed=cfg["seed"], split=cfg["eval_split"])
    cap = args.limit_corpus or cfg["corpus"].get("max_paras")
    texts, titles = build_corpus(split=cfg["corpus"]["split"], max_paras=cap)
    # gold doc_ids, by title. The corpus dedups on (title, text), so one title can still own
    # several doc_ids when the same page appears with different text across questions. Keep the
    # union: a hit on ANY of them is a hit on the supporting paragraph.
    title_to_ids: dict[str, list[int]] = collections.defaultdict(list)
    for i, t in enumerate(titles):
        title_to_ids[t].append(i)
    for r in rows:
        r["gold_ids"] = [i for t in r["gold_titles"] for i in title_to_ids.get(t, [])]
    missing = sum(1 for r in rows if not r["gold_ids"])
    corpus_info = {"split": cfg["corpus"]["split"], "paragraphs": len(texts),
                   "max_paras": cap, "eval_split": cfg["eval_split"], "n": n,
                   "queries_with_no_gold_in_corpus": missing}
    print(f"[{cid}] corpus {len(texts)} paragraphs | {n} queries | "
          f"{missing} queries whose gold is absent from the index", flush=True)
    if missing:
        print(f"[{cid}] ABORT: eval_split and corpus.split disagree, so gold paragraphs are "
              "outside the index and every retrieval metric would be meaningless. "
              "Set eval_split == corpus.split.", flush=True)
        return 2

    pricing = load_pricing(model_key)
    write_manifest(out_dir, cfg, pipeline_name, model_key, n, corpus_info, pricing)
    print(f"[{cid}] manifest written -> {out_dir/'manifest.json'}", flush=True)

    # ---- retrieval + correction (co-resident with the server from here on) ----
    from src.retrieval.hybrid_reranker import HybridLI

    hy = HybridLI(texts, w_sparse=cfg["retrieval"]["w_sparse"],
                  w_dense=cfg["retrieval"]["w_dense"],
                  beta_rerank=cfg["retrieval"]["beta_rerank"],
                  device="cuda", sparse=cfg["retrieval"]["sparse"],
                  cand_k=int(cfg["retrieval"]["cand_k"]))
    crag = None
    if pipeline.correct:
        from src.evaluator.crag_module import CRAGEvaluator

        crag = CRAGEvaluator(model_name=cfg["crag"]["model"],
                             upper_threshold=cfg["crag"]["upper"],
                             lower_threshold=cfg["crag"]["lower"],
                             strip_top_n=cfg["crag"]["strip_top_n"], device="cuda")

    # ---- serve + run ---------------------------------------------------------
    from src.runner.serve import (build_embeddings, build_judge, serve_client, serve_generate,
                                  server_from_config, serving)

    logger = MetricsLogger(out_dir / "queries.jsonl", pricing=pricing)
    t_wall = time.perf_counter()
    with serving(server_from_config(cfg, "generator")) as server:
        client = serve_client(server)

        def generate(prompt: str):
            return serve_generate(client, server.model, prompt,
                                  cfg["generator"]["max_new_tokens"],
                                  cfg["generator"].get("temperature", 0.0))

        results, runs = [], []
        for i, row in enumerate(rows):
            res = answer_query(pipeline, hy, crag, generate, row)
            # Label BEFORE logging so queries.jsonl carries it — the per-query record is the
            # committed raw data, and a label that only exists in the scorecard aggregate cannot
            # be re-derived from it once a model judge replaces these labels.
            res.extra["action"] = res.action
            res.extra["label"] = label(res.answer, row.get("answer", ""))
            # Rule 3 is enforced HERE, not in score_run: only the caller knows whether the gold
            # paragraphs were in the index, so only the caller can tell a real all-zero
            # retrieval from a query whose gold was never there to be found.
            res.extra["gold_in_index"] = bool(res.gold_ids)
            logger.log_query(
                query_id=res.query_id or f"q{i}", latency_ms=res.latency_ms,
                prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                model_name=cfg["models"][model_key]["model_id"],
                stage_latency_ms=res.stage_latency_ms, ttft_ms=res.ttft_ms,
                action=res.action, label=res.extra["label"], cell=cid,
            )
            results.append(res.to_dict())
            runs.append(score_run(res.retrieved_ids, res.gold_ids,
                                  ks=cfg["retrieval"]["metrics_ks"], depth=len(res.retrieved_ids)))
            if i % 5 == 0 or i == len(rows) - 1:
                print(f"[{cid}] {i+1}/{len(rows)} action={str(res.action):<9} "
                      f"label={res.extra['label']:<12} {res.latency_ms:7.0f}ms "
                      f"ttft={res.ttft_ms if res.ttft_ms is None else round(res.ttft_ms)}ms "
                      f"{res.answer[:60]!r}", flush=True)

        judge = build_judge(server, cfg["judge"]["mode"])
        embeddings = build_embeddings()
        try:
            ragas = ragas_scores(results, judge, embeddings)
        except Exception as e:      # never let ragas discard a completed generation pass
            ragas = {"error": f"{type(e).__name__}: {e}"}
    wall_s = time.perf_counter() - t_wall
    # GPU cost is amortised over the WHOLE cell: every query genuinely paid for the index build
    # and the server start, so per-query cost must too, not just the generation slice.
    if pricing.get("gpu_usd_per_hour"):
        per_q = wall_s / max(1, len(rows))
        for rec in logger.records:
            rec.gpu_cost_usd = per_q * pricing["gpu_usd_per_hour"] / 3600.0

    labels = [r["extra"]["label"] for r in results]
    summary = logger.summary()
    scorecard = {
        "cell": cid, "config": cfg, "summary": summary,
        "corpus": corpus_info, "pricing": pricing,
        "wall_clock_s": wall_s,
        "retrieval": aggregate(runs),
        "adjusted_accuracy_pct": adjusted_accuracy(labels),
        "adjusted_accuracy_formula": "Correct / (Correct + Hallucinated) * 100",
        "raw_accuracy_pct": 100.0 * labels.count("Correct") / len(labels),
        "hallucination_rate_pct": 100.0 * labels.count("Hallucinated") / len(labels),
        "rejection_rate_pct": 100.0 * labels.count("NoAnswer") / len(labels),
        "unlabelled_pct": 100.0 * labels.count(UNKNOWN) / len(labels),
        "label_note": "deterministic substring labels. 'Unknown' = no reference answer for that "
                      "row; excluded from adjusted accuracy and reported as unlabelled_pct "
                      "rather than counted as hallucination (contract rule 3)",
        "crag_action_distribution": dict(
            collections.Counter(r["action"] for r in results if r["action"])),
        "ragas": ragas,
        "queries": results,
    }
    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: scorecard[k] for k in
                      ("cell", "wall_clock_s", "adjusted_accuracy_pct", "rejection_rate_pct")},
                     indent=2))
    print("retrieval:", json.dumps(scorecard["retrieval"], indent=2))
    # `or 0.0` here was a rule-2 violation of this repo's own making: it printed $0.000000
    # whenever cost was unavailable, which is indistinguishable from a genuinely free query.
    # The contract scanner (src/contract.py) caught it; unavailable now prints as n/a.
    def _usd(v) -> str:
        return "n/a (unavailable)" if v is None else f"${v:.6f}"

    print(f"cost: api {_usd(summary['api_cost_usd'])} | gpu {_usd(summary['gpu_cost_usd'])} "
          f"(pricing {pricing['version']}, verified={pricing['verified']})")

    # ---- gates ----
    checks = phase_a_gates(pipeline, results, summary, scorecard, n, logger.records, runs)
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{sum(checks.values())}/{len(checks)} gates green -> {out_dir}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # Gate-logic self-check, no GPU and no network. A gate that has never been exercised is
        # a gate that will be trusted; this one is the difference between "the run was slow" and
        # "the run was dishonest", so it must be known to actually fire.
        n = 3
        stages = {"retrieve": 5.0, "rerank": 7.0, "crag": 9.0, "generate": 100.0}
        results = [{"answer": "a", "latency_ms": sum(stages.values()) + 4.0,
                    "stage_latency_ms": stages, "ttft_ms": 40.0, "retrieved_ids": [1, 2]}]
        summary = {"n_queries": n, "latency_ms": {"p95": 200.0},
                   "stage_latency_ms": {k: {"mean": 1.0, "p95": 2.0, "n": n} for k in stages},
                   "api_cost_usd": 0.0004}
        card = {"adjusted_accuracy_pct": 50.0, "ragas": {"faithfulness_n": n},
                "crag_action_distribution": {"Correct": 1, "Ambiguous": 1, "Incorrect": 1}}
        # Config/code drift check. `retrieval.top_k` and `pipelines` were both in the YAML and
        # read by nothing, so editing them did nothing — the config described runs that never
        # happened. Assert every key the code actually reads exists, so a rename is loud.
        cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
        required = {
            "": ("n", "seed", "eval_split", "output_root", "corpus", "retrieval", "crag",
                 "generator", "judge", "serve", "models"),
            "corpus": ("split", "max_paras"),
            "retrieval": ("sparse", "w_sparse", "w_dense", "beta_rerank", "top_k", "cand_k",
                          "metrics_ks"),
            "crag": ("model", "upper", "lower", "strip_top_n"),
            "generator": ("max_new_tokens", "temperature"),
            "judge": ("mode", "serve_id"),
            "serve": ("host", "port", "gpu_memory_utilization", "max_model_len"),
        }
        for path, keys in required.items():
            block = cfg[path] if path else cfg
            missing = [k for k in keys if k not in block]
            assert not missing, f"configs/experiment.yaml {path or '<root>'}: missing {missing}"
        for key, m in cfg["models"].items():
            assert {"model_id", "serve_id"} <= m.keys(), f"model {key}: need model_id + serve_id"
        assert cfg["eval_split"] == cfg["corpus"]["split"], (
            "locked decision 6: eval and corpus must share a split, or gold is outside the index")
        assert cfg["retrieval"]["sparse"] in ("pyserini", "rank_bm25"), cfg["retrieval"]["sparse"]
        assert len(cfg["models"]) == 3, f"expected 3 models, got {sorted(cfg['models'])}"

        p = PIPELINES["crag"]
        run_scores = [{"depth": 10.0, "map@3": 0.5, "coverage": 1.0}]
        from src.contract import UNKNOWN, scan_source

        # A deliberately fabricated cost, built through a named local so this file still passes
        # its own rule-2 scan: writing the literal here would be the violation the scanner hunts.
        free = 0.0

        recs = [QueryRecord(query_id="q", model_name="m",
                            latency_ms=sum(stages.values()) + 4.0, prompt_tokens=100,
                            completion_tokens=20, stage_latency_ms=stages, ttft_ms=40.0,
                            api_cost_usd=0.0001, gpu_cost_usd=0.00001)]
        res0 = [{**results[0], "gold_ids": [1, 2],
                 "extra": {"label": "Correct", "gold_in_index": True}}]
        gates = phase_a_gates(p, res0 * n, summary, card, n, recs, run_scores * n)
        assert all(gates.values()), [k for k, v in gates.items() if not v]

        # Each gate must fire on its own failure — assert the check is load-bearing, not vacuous.
        def g(res=res0, s=summary, c=card, rc=recs, rn=run_scores):
            return phase_a_gates(p, res * n, s, c, n, rc * n, rn * n)

        def clone(rows):
            return [{**r, "extra": dict(r["extra"]),
                     "stage_latency_ms": dict(r["stage_latency_ms"])} for r in rows]

        for name, break_it in (
            ("observed latency dominates stages",
             lambda r: [r.update(latency_ms=1.0)]),                      # derived, not observed
            ("TTFT measured on every query", lambda r: [r.update(ttft_ms=None)]),
            ("all answers non-empty", lambda r: [r.update(answer="  ")]),
            ("retrieval depth recorded", lambda r: [r.update(retrieved_ids=[])]),
            ("contract: no fabricated verdict",
             lambda r: [r["extra"].update(label=UNKNOWN)]),
        ):
            bad = clone(res0)
            break_it(bad[0])
            assert not g(bad)[name], f"gate {name!r} did not fire"
        # Stage-presence gates read the summary (which MetricsLogger derives from the same
        # records), so they are broken at the summary.
        for name, drop in (("rerank is its own stage (reranked/crag only)", "rerank"),
                           ("crag is its own stage (crag only)", "crag")):
            s2 = {**summary, "stage_latency_ms": {k: v for k, v in summary["stage_latency_ms"].items()
                                                  if k != drop}}
            assert not g(s=s2)[name], f"gate {name!r} did not fire"
        for name, card_bad in (("ragas faithfulness scored", {"ragas": {}}),
                               ("adjusted accuracy reported", {"adjusted_accuracy_pct": None})):
            assert not g(c={**card, **card_bad})[name], name
        # summary-level gates
        for name, s_bad in (("cost is not zero", {"api_cost_usd": free}),
                            ("P95 latency present", {"latency_ms": {"p95": None}}),
                            ("queries.jsonl complete", {"n_queries": 1})):
            assert not g(s={**summary, **s_bad})[name], name
        # An all-"Correct" CRAG run is a red flag, not a success: the evaluator never fired.
        assert not g(c={**card, "crag_action_distribution": {"Correct": 3}})[
            "CRAG has a non-Correct trigger"]
        # contract gates, broken at their own inputs
        for name, rc in (("contract: no fabricated measurement",
                          [replace(recs[0], api_cost_usd=free)]),
                         ("contract: no fabricated measurement",
                          [replace(recs[0], latency_ms=1.0)])):
            assert not g(rc=rc)[name], f"gate {name!r} did not fire on {rc[0]}"
        assert not g(rn=[{"map@3": 0.5}])["contract: retrieval scores carry metadata"], \
            "a score with no depth must fail the gate"

        # baseline has no rerank/crag stage, so those two gates must not demand them.
        b = phase_a_gates(PIPELINES["baseline"], res0 * n, summary, card, n, recs * n,
                          run_scores * n)
        assert b["rerank is its own stage (reranked/crag only)"] and b["crag is its own stage (crag only)"]
        assert b["contract: source has no rule-2 violation"], \
            scan_source("src") and "rule-2 violation in src/"
        print(f"run_cell gates OK: {len(gates)} gates, each verified to fire")
        sys.exit(0)
    sys.exit(main())
