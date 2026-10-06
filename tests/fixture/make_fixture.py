"""Build a tiny fake results/ tree: 2 cells × 20 queries each, so the scorecard/report layer has something to
render without a GPU. Deterministic by construction -> scorecard math on it is hand-computable.
"""
import json
import pathlib

root = pathlib.Path(__file__).parent / "results"
for cell, p95, faith, adj, action_mix in (
    ("baseline-qwen2.5-1.5b", 900.0, 0.75, 62.0, {"Correct": 14, "Ambiguous": 4, "Incorrect": 2}),
    ("crag-qwen2.5-1.5b",     1200.0, 0.80, 58.0, {"Correct": 17, "Ambiguous": 2, "Incorrect": 1}),
):
    p95 = p95 if False else p95  # no-op, keep values readable above
    lat = [p95 * 0.8, p95 * 0.9, p95, p95 * 1.1] * 5
    ttft = [l / 20 for l in lat]
    recs = []
    for i in range(20):
        recs.append({
            "question_id": f"q{i}", "answer": "a", "latency_ms": lat[i],
            "ttft_ms": ttft[i], "retrieved_ids": [1], "gold_ids": [1],
            "action": "Correct", "stage_latency_ms": {"retrieve": 3.0, "rerank": 4.0,
                                                       "generate": lat[i] - 7.0},
            "extra": {"label": "Correct" if i % 5 else "Hallucinated", "gold_in_index": True},
        })
    import math
    card = {
        "cell": cell, "config": {}, "summary": {
            "n_queries": 20,
            "latency_ms": {"mean": sum(lat) / 20, "p95": p95},
            "stage_latency_ms": {s: {"mean": lat[0] / 3, "p95": p95 / 3, "n": 20}
                                 for s in ("retrieve", "rerank", "generate")},
            "ttft_ms": {"mean": sum(ttft) / 20, "p95": max(ttft)},
            "tok_per_s": {"mean": 12.0, "p95": 10.0},
            "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
            "api_cost_usd": 0.0004, "gpu_cost_usd": 0.001,
            "n_with_api_cost": 20, "n_with_gpu_cost": 20},
        "corpus": {}, "pricing": {"version": "fixture", "verified": True},
        "wall_clock_s": 120.0,
        "retrieval": {"map@3": 0.9, "recall@10": 0.95},
        "adjusted_accuracy_pct": adj,
        "adjusted_accuracy_formula": "Correct / (Correct + Hallucinated) * 100",
        "raw_accuracy_pct": 80.0,
        "hallucination_rate_pct": 20.0,
        "rejection_rate_pct": 0.0, "unlabelled_pct": 0.0,
        "label_note": "fixture", "crag_action_distribution": action_mix,
        "ragas": {"faithfulness": faith, "faithfulness_n": 20},
        "queries": recs,
        "judge_serve_id": "j", "generator_serve_id": cell,
        "gates": {}, "gates_green": "17/17",
    }
    d = root / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("wrote", d / "scorecard.json")
