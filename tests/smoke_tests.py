"""3 assert-based CPU tests, no framework. Run: python -m tests.smoke_tests"""
from __future__ import annotations

import pathlib
import tempfile

from src.pipeline import PIPELINES, answer_query, QueryResult


def _false(name):
    raise AssertionError(name)


def test_pipeline_ablation_changes_contexts():
    """crag must actually route through the evaluator (so contexts differ when the evaluator says
    Incorrect), while baseline must pass the retrieved contexts straight to generation."""
    captured = {}

    class Hy:
        def retrieve(self, q):
            return [{"text": "the passage", "doc_id": 1}]

        def rerank(self, q, docs, k):
            return docs[:k]

    class Crag:
        calls = 0

        def evaluate(self, q, docs):
            self.calls += 1
            return {"action": "Incorrect", "strips": []}

    def generate(prompt):
        captured["prompt"] = prompt
        return ("ans", 10, 5, 12.0)

    crag = Crag()
    row = {"question": "q", "question_id": "q0", "answer": "a"}

    crag.calls, captured["prompt"] = 0, None
    answer_query(PIPELINES["crag"], Hy(), crag, generate, row)
    crag_prompt = captured["prompt"]
    assert crag.calls == 1, "crag pipeline must call the evaluator"
    assert "the passage" not in crag_prompt, "Incorrect must strip context"

    answer_query(PIPELINES["baseline"], Hy(), crag, generate, row)
    assert crag.calls == 1, "baseline must NOT call the evaluator"
    assert "the passage" in captured["prompt"], "baseline passes contexts through"


def test_index_cache_round_trips():
    """Second load of a cached corpus must return the same retrieval ranking — the cache key must
    track corpus content, so a mutated corpus can never hit a stale index."""
    from src.retrieval.retriever import Retriever

    corpus = [f"passage {i} about topic {i % 3}" for i in range(6)]
    with tempfile.TemporaryDirectory() as tmp:
        a = Retriever(corpus, device="cpu", cache_dir=tmp)
        a_ids = [d["doc_id"] for d in a.retrieve("topic 0", k=3)]
        b = Retriever(corpus, device="cpu", cache_dir=tmp)
        assert a_ids == [d["doc_id"] for d in b.retrieve("topic 0", k=3)], "cache must round-trip"
        c = Retriever(corpus + ["a new passage"], device="cpu", cache_dir=tmp)
        assert c.root != a.root, "corpus change must re-key the cache dir"


def test_scorecard_math_on_fixture():
    """The fixture tree is hand-computed: baseline adj_acc 62, faithfulness 0.75, crag 0.80, so
    Pareto_vs_latency must keep both, Pareto_vs_cost drops baseline (same cost, worse score)."""
    from analysis.scorecard import load_cells, cell_row, pareto

    cells = load_cells(pathlib.Path("tests/fixture/results"))
    rows = [cell_row(c) for c in cells.values()]
    assert len(rows) == 2, rows

    f_lat = {r["cell"] for r in pareto(rows, "lat_p95_ms", "faithfulness", True, True)}
    assert f_lat == {"baseline-qwen2.5-1.5b", "crag-qwen2.5-1.5b"}, f_lat

    f_cost = {r["cell"] for r in pareto(rows, "api_usd", "faithfulness", True, True)}
    assert f_cost == {"crag-qwen2.5-1.5b"}, f_cost

    by_cell = {r["cell"]: r for r in rows}
    assert by_cell["baseline-qwen2.5-1.5b"]["faithfulness"] == 0.75
    assert by_cell["crag-qwen2.5-1.5b"]["lat_p95_ms"] == 1200.0


TESTS = [test_pipeline_ablation_changes_contexts,
         test_index_cache_round_trips,
         test_scorecard_math_on_fixture]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(TESTS)}/{len(TESTS)} passed")
