"""Scorecard + Pareto fronts + pairwise stats over results/*/scorecard.json.

n=300 and 6 cells: two cells among the same queries are Paired by query_id. Wilcoxon signed-rank
for the continuous per-query metric (latency); McNemar for the binary per-query Correct label
(Wilcoxon on binary differences degenerates to a sign test, so correctness uses McNemar).
BH-FDR across all reported comparisons. No live GPU needed — this reads committed JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

def _scipy_p(values: list[tuple[float, float]]) -> float | None:
    """Wilcoxon signed-rank p on paired diffs; None when it cannot be computed (all-zero diff)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    diffs = [a - b for a, b in values if a != b]
    if len(diffs) < 10:
        return None
    return float(wilcoxon(diffs, zero_method="wilcox").pvalue)


def load_cells(root: Path) -> dict[str, dict]:
    cells = {}
    for p in sorted(root.glob("*/scorecard.json")):
        card = json.loads(p.read_text(encoding="utf-8"))
        cells[card["cell"]] = card
    return cells


def cell_row(card: dict) -> dict:
    s = card["summary"]
    ret = card.get("retrieval", {})
    ragas = card.get("ragas") or {}
    return {
        "cell": card["cell"],
        "n": s["n_queries"],
        "adj_acc_%": card.get("adjusted_accuracy_pct"),
        "faithfulness": ragas.get("faithfulness"),
        "faith_n": ragas.get("faithfulness_n"),
        "lat_p95_ms": s["latency_ms"]["p95"],
        "ttft_p95_ms": s["ttft_ms"]["p95"],
        "api_usd": s.get("api_cost_usd"),
        "gpu_usd": s.get("gpu_cost_usd"),
        "map@3": ret.get("map@3"),
        "recall@10": ret.get("recall@10"),
        "crag_dist": card.get("crag_action_distribution"),
    }


def pareto(rows: list[dict], x: str, y: str, x_min: bool, y_max: bool,
           key_exclude=None) -> list[dict]:
    """Row is on the front if no other row is at least as good on both axes and better on one."""
    front = []
    for r in rows:
        if r[x] is None or r[y] is None:
            continue
        dominated = False
        for o in rows:
            if o is r or o[x] is None or o[y] is None:
                continue
            o_x_better_or_eq = o[x] <= r[x] if x_min else o[x] >= r[x]
            o_y_better_or_eq = o[y] >= r[y] if y_max else o[y] <= r[y]
            o_strict = o[x] < r[x] if x_min else o[x] > r[x]
            o_strict_y = o[y] > r[y] if y_max else o[y] < r[y]
            if o_x_better_or_eq and o_y_better_or_eq and (o_strict or o_strict_y):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def bh_fdr(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [float("inf")] * m
    running = 1.0
    for rank_pos, i in reversed(list(enumerate(order, 1))):
        running = min(running, p_values[i] * m / rank_pos)
        adjusted[i] = min(running, 1.0)
    return adjusted


def per_query_correct(card: dict) -> dict[str, int]:
    return {q.get("question_id", f"i{i}"): int((q.get("extra") or {}).get("label") == "Correct")
            for i, q in enumerate(card.get("queries", []))}


def per_query_latency(card: dict) -> dict[str, float]:
    return {q.get("question_id", f"i{i}"): q.get("latency_ms")
            for i, q in enumerate(card.get("queries", []))
            if q.get("latency_ms") is not None}


def cell_pairs(cells: dict[str, dict]) -> list[tuple[str, str]]:
    """Pair crag <-> baseline within each model: that's the ablation the thesis interprets."""
    by_model: dict[str, set[str]] = defaultdict(set)
    for cid in cells:
        pipeline, _, model = cid.partition("-")[0], None, None
        # cell ids are '<pipeline>-<model_key>', model_key carries dashes -> split once.
        pipeline, model = cid.split("-", 1)
        by_model[model].add(pipeline)
    return [(f"crag-{m}", f"baseline-{m}") for m, pipes in by_model.items()
            if {"crag", "baseline"} <= pipes]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar over discordant queries b (crag correct, baseline wrong) and
    c (baseline correct, crag wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(2 * cdf, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="scorecard roll-up + Pareto +stats")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/scorecard_summary.md")
    a = ap.parse_args()

    cells = load_cells(Path(a.results))
    rows = [cell_row(c) for c in cells.values()]

    lines = ["| cell | n | adj acc % | faithfulness | lat p95 ms | ttft p95 ms | api $ | gpu $ | MAP@3 | recall@10 | crag dist |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {cell} | {n} | {adj_acc_%:.1f} | {faithfulness} | {lat_p95_ms:.0f} | {ttft_p95_ms:.1f} | {api_usd:.4f} | {gpu_usd:.4f} | {map@3:.3g} | {recall@10:.3g} | {crag_dist} |".format(**{**r, "faithfulness": r["faithfulness"] if r["faithfulness"] is not None else "n/a"}))
    print("\n".join(lines))

    fronts = {
        "faithfulness_vs_latency": pareto(rows, "lat_p95_ms", "faithfulness", x_min=True, y_max=True),
        "faithfulness_vs_total_cost": pareto(rows, "api_usd", "faithfulness", x_min=True, y_max=True),
    }
    print("\nPareto fronts:")
    for name, front in fronts.items():
        print(f"  {name}: {[r['cell'] for r in front]}")

    # Stats over the crag-vs-baseline cell pairs.
    raw_p, labels = [], []
    for crag_id, base_id in cell_pairs(cells):
        crag_c, base_c = per_query_correct(cells[crag_id]), per_query_correct(cells[base_id])
        common = sorted(set(crag_c) & set(base_c))
        b = sum(crag_c[q] == 1 and base_c[q] == 0 for q in common)
        c = sum(crag_c[q] == 0 and base_c[q] == 1 for q in common)
        crag_lat, base_lat = per_query_latency(cells[crag_id]), per_query_latency(cells[base_id])
        w = _scipy_p([(crag_lat[q], base_lat[q]) for q in common if q in crag_lat and q in base_lat])
        if w is not None:
            raw_p.append(w)
            labels.append(f"{crag_id} vs {base_id} round-trip latency")
        raw_p.append(mcnemar_p(b, c))
        labels.append(f"{crag_id} vs {base_id} correctness (McNemar)")
    adjusted = bh_fdr(raw_p)
    print("\npaired crag-vs-baseline (BH-FDR):")
    for l, p, q in zip(labels, raw_p, adjusted):
        print(f"  p={p:.3g} q_fdr={q:.3g}  {l}")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n\nPareto:\n" +
                        "\n".join(f"- {k}: {[r['cell'] for r in v]}" for k, v in fronts.items()) +
                        "\n\nStats:\n" +
                        "\n".join(f"- {l}: p={p:.3g}, q_fdr={q:.3g}" for l, p, q in zip(labels, raw_p, adjusted))
                        + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
