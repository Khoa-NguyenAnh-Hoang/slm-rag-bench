"""The pipeline spec and the timed chain.

A pipeline is a spec, not a code path: `baseline` and `crag` are two `Pipeline` values sharing
one `answer_query()`. Adding an ablation means adding a value, not a branch.

The reranker is NOT a pipeline axis. It is part of the fixed retriever (src/retrieval/), so it
is on in every cell and there is nothing to compare it against. That is deliberate: the
benchmark's question is which *model* to pick under a fixed retriever, and a retriever that
varies between cells makes two cells incomparable. See src/retrieval/NOTES.md for the
measurements that settled it.

One outer timer spans the whole chain and each stage keeps its own, so `latency_ms` is observed
rather than derived. `latency_ms >= sum(stage_latency_ms)` then holds by construction, and any
future re-batching surfaces as a gate failure instead of a quietly wrong number.

`generate` is injected, so the chain never imports vLLM, ragas or torch and the whole pipeline is
testable on a laptop with fakes. That is what keeps the matrix honest: a stage regression is
caught by a CPU assert instead of after a multi-hour run.
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
    correct: bool
    top_k: int = 10

    def stages(self) -> list[str]:
        return [s for s, on in (("retrieve", True), ("rerank", True),
                                ("crag", self.correct), ("generate", True)) if on]


PIPELINES: dict[str, Pipeline] = {p.name: p for p in (
    Pipeline("baseline", correct=False),
    Pipeline("crag",     correct=True),
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
    """Chain-of-thought prompt with an explicit abstention branch when context is insufficient."""
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

    hy    : Retriever (or a fake with .retrieve/.rerank)
    crag  : CRAGEvaluator (or a fake with .evaluate), ignored when p.correct is False
    generate : (prompt) -> (text, prompt_tokens, completion_tokens, ttft_ms|None)
    """
    t0 = time.perf_counter()
    st: dict[str, float] = {}

    with stage_timer(st, "retrieve"):
        docs = hy.retrieve(row["question"])
    with stage_timer(st, "rerank"):
        docs = hy.rerank(row["question"], docs, k=p.top_k)

    action: str | None = None
    cr: dict | None = None
    contexts = [d["text"] for d in docs]
    if p.correct:
        with stage_timer(st, "crag"):
            cr = crag.evaluate(row["question"], contexts)
        action = cr["action"]
        # Refinement only for Correct/Ambiguous. Incorrect => no internal knowledge => empty
        # context => the abstention branch of build_prompt. No live web search to rescue it ($0
        # constraint), so Incorrect can only be answered from parametric memory or not at all.
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
    """Deterministic Correct / Hallucinated / NoAnswer / Unknown labels.

    Deterministic string matching, not a model judge — no third-party judge is reachable under
    the $0 constraint, and an LLM judge replaces this in a later phase (src/monitor/labels.py).
    Kept because the hand-labelled agreement sample needs a reproducible reference to score
    against, and because yes/no HotpotQA answers are substring-matched exactly (a `gold in
    answer` test alone is meaningless for "yes"/"no").

    UNKNOWN is a fourth state, and it is load-bearing (contract rule 3). Without it an empty
    `gold` fell through to `Hallucinated` — charging the model with hallucinating because *we*
    lack the reference answer. That inflates the hallucination rate and depresses adjusted
    accuracy: both headline numbers, both wrong, both in the direction that flatters the thesis.
    `Unknown` is excluded from adjusted accuracy's denominator (see adjusted_accuracy), and
    `run_cell` fails the run if any query is Unknown.
    """
    import re

    from src.contract import UNKNOWN

    g = gold.strip().lower().rstrip(".")
    if not g:
        return UNKNOWN
    a = answer.strip().lower()
    if ABSTAIN_TEXT.lower() in a or re.search(
            r"does not provide sufficient information|cannot answer|not enough information", a):
        return "NoAnswer"
    if g in ("yes", "no"):
        return "Correct" if re.search(rf"\b{re.escape(g)}\b", a) else "Hallucinated"
    return "Correct" if g in a else "Hallucinated"


def adjusted_accuracy(labels: list[str]) -> float | float:
    """Correct / (Correct + Hallucinated) × 100 — abstentions excluded from both sides.

    Returns None when nothing was attempted — a model that abstains on everything has no
    adjusted accuracy, and that is a real outcome, not 0%. `Unknown` labels (no reference
    answer, contract rule 3) are excluded from the denominator for the same reason: they are
    not attempts, so counting them as hallucinations would penalise the model for a gap in
    our data. The count is reported separately so the exclusion is visible.
    """
    c = labels.count("Correct")
    h = labels.count("Hallucinated")
    return 100.0 * c / (c + h) if (c + h) else None


if __name__ == "__main__":
    # Self-check with fakes: no torch, no GPU, no network. This is the contract every cell obeys.
    class FakeHy:
        """retrieve() returns ASCENDING sparse score; rerank() returns DESCENDING. So the
        doc-id order provably differs between the two stages — if answer_query ignored
        rerank's output, retrieved_ids would come back ascending."""

        def __init__(self):
            self.calls = []

        def retrieve(self, q, k=None):
            self.calls.append(("retrieve", k))
            return [{"doc_id": i, "text": f"d{i}", "sparse_score": 0.1 * i} for i in range(6)]

        def rerank(self, q, docs, k=10):
            self.calls.append(("rerank", k))
            out = sorted(docs, key=lambda d: -d["sparse_score"])
            return [{"doc_id": d["doc_id"], "text": d["text"], "sparse_score": d["sparse_score"],
                     "final_score": d["sparse_score"]} for d in out][:k]

    class FakeCrag:
        def __init__(self, action):
            self.action = action

        def evaluate(self, q, docs):
            return {"action": self.action,
                    "strips": docs[:5] if self.action != "Incorrect" else []}

    def fake_gen(prompt):
        return ("answer text", 120, 30, 55.0)

    row = {"question_id": "q1", "question": "who?", "gold_ids": [2, 3]}

    # The reranker is part of the fixed retriever, so it runs in EVERY cell. baseline and crag
    # must therefore agree on retrieval and differ only in the correction stage.
    hy = FakeHy()
    r = answer_query(PIPELINES["baseline"], hy, None, fake_gen, row)
    assert [c for c, _ in hy.calls] == ["retrieve", "rerank"], hy.calls
    assert hy.calls[1][1] == 10, "rerank must cut the pool to the generator's top_k"
    assert r.action is None and len(r.contexts) == 6
    assert r.retrieved_ids == [5, 4, 3, 2, 1, 0], r.retrieved_ids
    assert set(r.stage_latency_ms) == {"retrieve", "rerank", "generate"}
    assert r.latency_ms >= sum(r.stage_latency_ms.values()), "outer timer must dominate stages"
    assert "reranked" not in PIPELINES, "the reranker is fixed, not a pipeline axis"
    assert PIPELINES["baseline"].stages() == ["retrieve", "rerank", "generate"]
    assert PIPELINES["crag"].stages() == ["retrieve", "rerank", "crag", "generate"]

    # crag: strips when Correct, empty context (abstention branch) when Incorrect
    r3 = answer_query(PIPELINES["crag"], FakeHy(), FakeCrag("Correct"), fake_gen, row)
    assert r3.action == "Correct" and len(r3.contexts) == 5 and "crag" in r3.stage_latency_ms
    assert r3.retrieved_ids == r.retrieved_ids, "same retriever => same docs in both pipelines"
    r4 = answer_query(PIPELINES["crag"], FakeHy(), FakeCrag("Incorrect"), fake_gen, row)
    assert r4.action == "Incorrect" and r4.contexts == []
    assert "no relevant context retrieved" in build_prompt(r4.question, r4.contexts)

    assert label("Yes, it was.", "yes") == "Correct"
    assert label("No it was not.", "yes") == "Hallucinated"
    assert label(ABSTAIN_TEXT, "yes") == "NoAnswer"
    assert label("The answer is Paris.", "paris") == "Correct"
    # Rule 3: no reference answer => Unknown, never a fabricated verdict. Before this, an empty
    # gold fell through to "Hallucinated" and charged the model for our missing data.
    assert label("The answer is Paris.", "") == "Unknown", label("The answer is Paris.", "")
    assert label("anything", "   ") == "Unknown"
    assert adjusted_accuracy(["Correct", "Hallucinated", "NoAnswer"]) == 50.0
    # Unknown is not an attempt: excluded from the denominator, not charged as a hallucination.
    assert adjusted_accuracy(["Correct", "Hallucinated", "Unknown"]) == 50.0
    assert adjusted_accuracy(["NoAnswer", "NoAnswer"]) is None
    assert adjusted_accuracy(["Unknown", "Unknown"]) is None
    print("pipeline OK:", len(PIPELINES), "pipelines, stage sets",
          {n: p.stages() for n, p in PIPELINES.items()})
