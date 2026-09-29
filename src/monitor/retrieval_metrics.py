"""Retrieval quality metrics: MAP@k, NDCG@k, Recall@k, MRR, Coverage.

Justification: EvaRAG (Elkiran & Rasheed, 2025) §2.1 logs Recall@k, MRR, nDCG@k and Coverage
as its retrieval layer; Action Plan Task 2's success criterion is "MAP@3 and NDCG@10 match
Mala et al. baseline trends". Neither existed in the MVP — it measured no retrieval quality at
all, so the reranking ablation had nothing to separate on.

AP@k and NDCG@k are ported from the reference implementation
(Beyond_Retrieval-main/02_hybrid_pipeline/hybrid_pipeline.ipynb:327-370, `average_precision_at_k_
from_binary` / `dcg_at_k_from_binary` / `ndcg_at_k_from_binary`) so the numbers are computed the
same way the paper computes them.

DEPTH IS PART OF THE METRIC — read this before comparing to Mala's 0.2819.
The reference `num_relevant = sum(rel)` counts relevant documents found within the retrieved
list, NOT in the whole corpus. So the AP denominator grows with retrieval depth: retrieve 10
deep and find 1 gold and you score AP@3 = 1.0; retrieve 200 deep, find 4 gold, and the same hit
scores 0.25. Every call therefore takes an explicit `depth`, and `score_run` records it in its
output, because a MAP@3 quoted without its depth is not comparable to anything.

Mala's exact corpus is not vendored (their notebook reads `hotpotqa_fulldataset_cleaned.csv`,
absent from this repo; the Phase 1 audit flagged it `[UNCERTAIN]`). So 0.2819 is a reported
COMPARISON, not a pass/fail gate — see Notes/Phase 2 Plan.md A5.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def rel_vector(retrieved_ids: Sequence[int], gold_ids: Iterable[int]) -> list[int]:
    """0/1 relevance aligned to the retrieved order. Gold is matched on paragraph identity."""
    gold = set(gold_ids)
    return [1 if d in gold else 0 for d in retrieved_ids]


def average_precision_at_k(rel: Sequence[int], k: int) -> float:
    """Ported from the reference impl: sum(precision@i where rel[i]=1) / num_relevant, where
    num_relevant counts gold found within the retrieved list (see module docstring)."""
    num_relevant = sum(rel)
    if num_relevant == 0:
        return 0.0
    hits, ap_sum = 0, 0.0
    for i, r in enumerate(rel[:k], start=1):
        if r == 1:
            hits += 1
            ap_sum += hits / i
    return ap_sum / num_relevant


def dcg_at_k(rel: Sequence[int], k: int) -> float:
    return sum(1.0 / math.log2(i + 1) for i, g in enumerate(rel[:k], start=1) if g)


def ndcg_at_k(rel: Sequence[int], k: int) -> float:
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg / idcg


def recall_at_k(rel: Sequence[int], k: int) -> float:
    """Fraction of gold actually found in the retrieved list that was retrieved. Unlike AP this
    is depth-independent, which makes it the right gate for the rerank-vs-baseline comparison."""
    total = sum(rel)
    return 0.0 if total == 0 else sum(rel[:k]) / total


def reciprocal_rank(rel: Sequence[int]) -> float:
    for i, r in enumerate(rel, start=1):
        if r == 1:
            return 1.0 / i
    return 0.0


def score_run(
    retrieved_ids: Sequence[int],
    gold_ids: Iterable[int],
    ks: Sequence[int] = (1, 3, 5, 10),
    depth: int | None = None,
) -> dict[str, float]:
    """Score one query. `depth` (how deep retrieval actually went) is recorded in the result so
    the MAP numbers are self-describing and cannot be compared across runs at different depths."""
    rel = rel_vector(retrieved_ids[: depth or len(retrieved_ids)], gold_ids)
    out: dict[str, float] = {"depth": float(depth or len(retrieved_ids))}
    for k in ks:
        out[f"map@{k}"] = average_precision_at_k(rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(rel, k)
    out["mrr"] = reciprocal_rank(rel)
    out["coverage"] = 1.0 if any(rel) else 0.0
    return out


def aggregate(runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean over queries. Non-zero-count metrics are averaged as-is; recall is already a ratio
    per query, so its mean is the micro-average over queries, not over gold documents."""
    if not runs:
        return {}
    keys = runs[0].keys()
    return {k: float(sum(r[k] for r in runs) / len(runs)) for k in keys}


if __name__ == "__main__":
    # Known-answer checks against hand-computed values.
    rel = [1, 0, 1, 0, 0]
    # gold at ranks 1 and 3: precision@1=1, precision@3=2/3 -> (1 + 2/3)/2 = 0.8333
    assert abs(average_precision_at_k(rel, 3) - (1 + 2 / 3) / 2) < 1e-9
    assert average_precision_at_k([0, 0, 0], 3) == 0.0            # no gold found
    assert ndcg_at_k([0, 0], 3) == 0.0                            # no gold
    assert recall_at_k(rel, 3) == 1.0 and recall_at_k(rel, 2) == 0.5
    assert abs(reciprocal_rank(rel) - 1.0) < 1e-9 and reciprocal_rank([0, 0, 1]) == 1 / 3

    s = score_run([7, 8, 9, 10], gold_ids=[7, 9], ks=(1, 3), depth=4)
    # rel = [1,0,1,0]; num_relevant = 2, so AP divides by 2 even at k=1: precision@1 = 1.0
    # becomes AP@1 = 0.5. This is the reference implementation's convention, not a bug.
    assert s["map@1"] == 0.5 and s["recall@1"] == 0.5 and s["mrr"] == 1.0
    assert abs(s["map@3"] - (1 + 2 / 3) / 2) < 1e-9, s
    # Depth sensitivity, the reason depth is a parameter: the same two hits at the same ranks,
    # but a deeper retrieval finds a third gold and the denominator grows.
    deep = score_run([7, 8, 9, 10, 11], gold_ids=[7, 9, 11], ks=(3,), depth=5)
    assert deep["map@3"] < s["map@3"], "AP must be depth-sensitive — that is the whole point"
    assert deep["recall@3"] == 2 / 3 and s["recall@3"] == 1.0
    assert aggregate([s, s])["map@1"] == 0.5 and aggregate([]) == {}
    print("retrieval_metrics OK:", {k: round(v, 3) for k, v in s.items()})
