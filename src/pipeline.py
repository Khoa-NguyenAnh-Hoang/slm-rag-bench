"""The pipeline spec and the timed chain — Phase 2 plan §2.2.

Why this file exists: the MVP's `mvp_runner.main()` hardcoded exactly one chain
(hybrid + rerank + CRAG -> generate), which made the Action Plan's central ablation
(baseline vs reranked vs CRAG) inexpressible. Worse, it reported
`latency_ms = sum(stage_latency_ms)` from a stage-batched run — a sum of separately measured
distributions, not an observed end-to-end number. Latency is the thesis's "+1", so that number
has to be real.

Three properties are load-bearing:

1. **A pipeline is a spec, not a code path.** Three `Pipeline` values, one `answer_query`.
2. **One outer timer, four stage timers.** `latency_ms` is measured around the whole chain, so
   the gate `latency_ms >= sum(stages)` holds by construction and any future re-batching shows up
   as a test failure rather than as a quietly wrong number.
3. **`generate` is injected.** The chain never imports vLLM, ragas or torch, so the whole
   pipeline is testable on a laptop with fakes (see the self-check below). That is what keeps
   the matrix runner honest: if a stage regresses, a CPU assert catches it before a 28-GPU-hour
   campaign does.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable

from src.monitor.metrics_logger import stage_timer

ABSTAIN_TEXT = "The context does not provide sufficient information to answer the question."


@dataclass(frozen=True)
class Pipeline:
    """Which stages are ON. Retrieval hyperparameters and the CRAG thresholds are NOT here —
    they live in configs/experiment.yaml and are read once by the runner, so every value has
    exactly one source of truth (a strip_top_n knob in two places is how a scorecard ends up
    describing a run that never happened)."""
    name: str
    rerank: bool
    correct: bool
    top_k: int = 10

    def stages(self) -> list[str]:
        return [s for s, on in (("retrieve", True), ("rerank", self.rerank),
                                ("crag", self.correct), ("generate", True)) if on]


PIPELINES: dict[str, Pipeline] = {p.name: p for p in (
    Pipeline("baseline", rerank=False, correct=False),
    Pipeline("reranked", rerank=True,  correct=False),
    Pipeline("crag",     rerank=True,  correct=True),
)}


@dataclass
class QueryResult:
    query_id: str
    question: str
    answer: str
    contexts: list[str]
    retrieved_ids: list[int]
    gold_ids: list[int] = field(default_factory=list)
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    action: str | None = None
    crag: dict | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gold_ids"] = self.gold_ids        # kept for retrieval_metrics; not in the report table
        return d


def build_prompt(question: str, contexts: list[str]) -> str:
    """CoT + abstention-inducing prompt, ported from Beyond_Retrieval generation_utils.py:120."""
    ctx = "\n".join(f"[context {i}] - {' '.join(c.split())[:2000]}"
                    for i, c in enumerate(contexts, 1)) or "[no relevant context retrieved]"
    return (
        "You are a helpful assistant. Answer strictly based on the provided contexts.\n\n"
        f"Question:- {question}\nContexts:-\n{ctx}\n\n"
        "Think step by step. If the contexts do not provide sufficient information, "
        f'respond with: "{ABSTAIN_TEXT}" '
        "Otherwise generate a grounded answer in one or two sentences.\nAnswer:"
    )


def answer_query(
    p: Pipeline,
    hy,
    crag,
    generate: Callable[[str], tuple[str, int, int, float | None]],
    row: dict,
) -> QueryResult:
    """Run one query end-to-end with a single outer timer.

    hy    : HybridLI (or a fake with .retrieve/.rerank)
    crag  : CRAGEvaluator (or a fake with .evaluate), ignored when p.correct is False
    generate : (prompt) -> (text, prompt_tokens, completion_tokens, ttft_ms|None)
    """
    t0 = time.perf_counter()
    st: dict[str, float] = {}

    with stage_timer(st, "retrieve"):
        docs = hy.retrieve(row["question"], k=p.top_k if not p.rerank else None)
    if p.rerank:
        with stage_timer(st, "rerank"):
            docs = hy.rerank(row["question"], docs, k=p.top_k)

    action: str | None = None
    cr: dict | None = None
    contexts = [d["text"] for d in docs]
    if p.correct:
        with stage_timer(st, "crag"):
            cr = crag.evaluate(row["question"], contexts)
        action = cr["action"]
        # Yan et al. 2C: refinement only for Correct/Ambiguous. Incorrect => no internal
        # knowledge => empty context => the abstention branch of build_prompt (Action Plan Risk 3,
        # $0 constraint: no live web search).
        contexts = cr["strips"] if action in ("Correct", "Ambiguous") else []

    with stage_timer(st, "generate"):
        text, p_tok, c_tok, ttft = generate(build_prompt(row["question"], contexts))

    return QueryResult(
        query_id=row.get("question_id", ""), question=row["question"], answer=text,
        contexts=contexts, retrieved_ids=[d["doc_id"] for d in docs],
        gold_ids=row.get("gold_ids", []), latency_ms=(time.perf_counter() - t0) * 1000.0,
        stage_latency_ms=st, action=action, crag=cr,
        prompt_tokens=p_tok, completion_tokens=c_tok, ttft_ms=ttft,
    )


def label(answer: str, gold: str) -> str:
    """Deterministic Correct/Hallucinated/NoAnswer labels (Mala et al. §E three-way protocol).

    Deterministic string matching, NOT the paper's claude-sonnet-4.5 judge — that is
    prohibited by the $0 constraint and is replaced by an LLM judge in Phase B
    (src/monitor/labels.py). Kept here because the hand-labelled agreement sample needs a
    reproducible reference, and because yes/no HotpotQA answers are substring-matched exactly
    (a `gold in answer` test alone is meaningless for "yes"/"no").
    """
    import re

    a = answer.strip().lower()
    if ABSTAIN_TEXT.lower() in a or re.search(
            r"does not provide sufficient information|cannot answer|not enough information", a):
        return "NoAnswer"
    g = gold.strip().lower().rstrip(".")
    if g in ("yes", "no"):
        return "Correct" if re.search(rf"\b{re.escape(g)}\b", a) else "Hallucinated"
    return "Correct" if (g and g in a) else "Hallucinated"


def adjusted_accuracy(labels: list[str]) -> float | None:
    """Mala et al. §E: Correct / (Correct + Hallucinated) × 100. None when nothing was attempted,
    which is a real outcome (a model that abstains on everything has no adjusted accuracy) and
    must not be reported as 0%."""
    c = labels.count("Correct")
    h = labels.count("Hallucinated")
    return 100.0 * c / (c + h) if (c + h) else None


if __name__ == "__main__":
    # Self-check with fakes: no torch, no GPU, no network. This is the contract every cell obeys.
    class FakeHy:
        """retrieve() returns ASCENDING fused score; rerank() returns DESCENDING. So the doc-id
        order provably differs between the two pipelines — if answer_query ignored rerank's
        output, retrieved_ids would come back ascending under the 'reranked' pipeline too."""

        def __init__(self):
            self.calls = []

        def retrieve(self, q, k=None):
            self.calls.append(("retrieve", k))
            return [{"doc_id": i, "text": f"d{i}", "fused_score": 0.1 * i} for i in range(6)]

        def rerank(self, q, docs, k=10):
            self.calls.append(("rerank", k))
            out = sorted(docs, key=lambda d: -d["fused_score"])
            return [{"doc_id": d["doc_id"], "text": d["text"], "fused_score": d["fused_score"],
                     "final_score": d["fused_score"]} for d in out][:k]

    class FakeCrag:
        def __init__(self, action):
            self.action = action

        def evaluate(self, q, docs):
            return {"action": self.action,
                    "strips": docs[:5] if self.action != "Incorrect" else []}

    def fake_gen(prompt):
        return ("answer text", 120, 30, 55.0)

    row = {"question_id": "q1", "question": "who?", "gold_ids": [2, 3]}

    # baseline: no rerank, no correction, full context
    hy = FakeHy()
    r = answer_query(PIPELINES["baseline"], hy, None, fake_gen, row)
    assert [c for c, _ in hy.calls] == ["retrieve"], hy.calls
    assert r.action is None and len(r.contexts) == 6
    assert r.retrieved_ids == [0, 1, 2, 3, 4, 5], r.retrieved_ids
    assert set(r.stage_latency_ms) == {"retrieve", "generate"}
    assert r.latency_ms >= sum(r.stage_latency_ms.values()), "outer timer must dominate stages"

    # reranked: rerank is its own stage, its own latency, and it actually reorders
    hy = FakeHy()
    r2 = answer_query(PIPELINES["reranked"], hy, None, fake_gen, row)
    assert [c for c, _ in hy.calls] == ["retrieve", "rerank"]
    assert set(r2.stage_latency_ms) == {"retrieve", "rerank", "generate"}
    assert r2.retrieved_ids == [5, 4, 3, 2, 1, 0], r2.retrieved_ids

    # crag: strips when Correct, empty context (abstention branch) when Incorrect
    r3 = answer_query(PIPELINES["crag"], FakeHy(), FakeCrag("Correct"), fake_gen, row)
    assert r3.action == "Correct" and len(r3.contexts) == 5 and "crag" in r3.stage_latency_ms
    r4 = answer_query(PIPELINES["crag"], FakeHy(), FakeCrag("Incorrect"), fake_gen, row)
    assert r4.action == "Incorrect" and r4.contexts == []
    assert "no relevant context retrieved" in build_prompt(r4.question, r4.contexts)

    assert label("Yes, it was.", "yes") == "Correct"
    assert label("No it was not.", "yes") == "Hallucinated"
    assert label(ABSTAIN_TEXT, "yes") == "NoAnswer"
    assert label("The answer is Paris.", "paris") == "Correct"
    assert adjusted_accuracy(["Correct", "Hallucinated", "NoAnswer"]) == 50.0
    assert adjusted_accuracy(["NoAnswer", "NoAnswer"]) is None
    print("pipeline OK: 3 pipelines, stage sets",
          [sorted(answer_query(PIPELINES[n], FakeHy(), FakeCrag("Correct"), fake_gen, row)
                   .stage_latency_ms) for n in ("baseline", "reranked", "crag")])
