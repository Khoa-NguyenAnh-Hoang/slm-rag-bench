"""The fixed retriever: bm25s sparse retrieval, then a MiniLM cross-encoder rerank.

`retrieve()` and `rerank()` stay split on purpose. Fused into one call the reranker's own
latency becomes invisible, which is exactly the quantity the scorecard reports, and the stage
timers in src/pipeline.py stop attributing time to the stage that spent it.

No dense arm. Measured on the whole 483,696-paragraph train corpus, sparse-only recall@30 is
0.9800 and the reranked top-10 is 0.9767 — so a dense arm fused into a 30-deep pool could
raise recall@10 by at most 0.0033. It cost a 438 MB model plus a tens-of-minutes encode, and
it made "which model" unanswerable by adding retrieval variance to every cell. Numbers and
method: src/retrieval/NOTES.md.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np

CAND_K = 30          # sparse candidate depth before rerank
DEFAULT_CACHE = "scratch/index"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def _tok(text: str) -> list[str]:
    """BM25 word tokens, and the ONLY tokeniser in the retrieval path.

    A whitespace split() glues punctuation onto the term ('paris?', '1879.'), so a query token
    almost never equals a document token: scores drift toward 0, the `> 0` filter empties the
    sparse arm, and the candidate pool comes back empty. It is passed to the index as explicit
    token lists rather than left to the backend's own tokeniser, so index and query cannot
    disagree about what a word is.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    """Sparse candidate pool, then cross-encoder rerank.

    cache_dir: building the index is worth persisting — ~42 s at 483,696 paragraphs here, and
    every cell would otherwise pay it again. The key is a hash of the corpus contents, so a
    different corpus can never hit a stale index.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        beta_rerank: float = 0.85,
        device: str = "cuda",
        cache_dir: str | Path = DEFAULT_CACHE,
        cand_k: int = CAND_K,
    ) -> None:
        self.beta = beta_rerank
        self.cand_k, self.device = cand_k, device
        self.corpus = list(corpus)

        # Lazy so the module body is importable on a box with no torch.
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(RERANKER, device=device)

        self.root = Path(cache_dir) / hashlib.md5(
            "\n".join(self.corpus).encode()).hexdigest()[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self._build_sparse()
        self.sparse_ms = 0.0

    # ---------- index construction (cached) ----------

    def _build_sparse(self) -> None:
        """bm25s index over the corpus, cached under a corpus-hash directory.

        `method="lucene"` rather than the wheel default: "robertson" floors IDF at zero, so a
        query of only common terms scores 0.0 against every document, the `> 0` filter drops
        them all, and the pool comes back empty — the benchmark then reports a retrieval
        failure caused by the backend. "lucene" and "atire" are identical on real queries.
        """
        import bm25s

        cache = self.root / "bm25s"
        # Our own marker, not a guess at the wheel's internal filenames: a save interrupted
        # partway leaves some files present and the rest missing, and load() would then fail
        # deep inside scipy instead of rebuilding.
        if (cache / "index.done").exists():
            # load() is a @classmethod that RETURNS a new object; it does not mutate in place.
            self.bm = bm25s.BM25.load(str(cache), method="lucene", show_progress=False)
        else:
            t0 = time.perf_counter()
            self.bm = bm25s.BM25(method="lucene")
            self.bm.index([_tok(d) for d in self.corpus], show_progress=False)
            self.bm.save(str(cache), show_progress=False)
            (cache / "index.done").touch()
            print(f"[retriever] indexed {len(self.corpus)} paragraphs in "
                  f"{time.perf_counter() - t0:.1f}s -> {cache}", flush=True)
        assert self.bm.scores is not None, "sparse index loaded without a score matrix"

    # ---------- stage 1: sparse candidate pool ----------

    def _sparse(self, query: str, k: int) -> dict[int, float]:
        """{doc_id: score}. Keys index `corpus` directly, in the same integer space throughout.

        The `> 0` filter is load-bearing, not cosmetic: an out-of-vocabulary term scores 0.0
        against every document, and without the filter a query with no lexical overlap returns
        k arbitrary documents that then get reranked as if they were candidates.
        """
        q = _tok(query)
        if not q:
            return {}
        scores = self.bm.get_scores(q)
        top = np.argsort(-scores)[:k]
        return {int(i): float(scores[i]) for i in top if scores[i] > 0}

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Stage 1 — the candidate pool, NO cross-encoder. Up to `k` docs (default cand_k) by
        BM25 score, each carrying `sparse_score` and no `ce_score`."""
        k = k or self.cand_k
        t0 = time.perf_counter()
        sp = self._sparse(query, k)
        self.sparse_ms = (time.perf_counter() - t0) * 1000
        if not sp:
            return []
        order = sorted(sp, key=sp.get, reverse=True)[:k]
        sc = _minmax(np.array([sp[i] for i in order]))
        return [{"doc_id": i, "text": self.corpus[i], "sparse_score": float(sc[j])}
                for j, i in enumerate(order)]

    # ---------- stage 2: cross-encoder rerank ----------

    def rerank(self, query: str, docs: list[dict], k: int = 10) -> list[dict]:
        """Stage 2 — MiniLM over `docs`, blended beta*CE + (1-beta)*BM25, return top k.

        The BM25 term is a tie-breaker between documents the cross-encoder finds equally
        relevant; the cross-encoder is what moves items into the top 10 (recall@10 0.9400 ->
        0.9767 at full corpus scale).

        Full text is passed: the CrossEncoder truncates to its own 512-token limit internally. A
        character slice fed the ranker ~120 tokens while the returned context was the whole
        paragraph, so ce_score and the emitted text disagreed.
        """
        if not docs:
            return []
        pairs = [(query, d["text"]) for d in docs]
        ce = _minmax(np.asarray(self.reranker.predict(pairs), dtype=float))
        sp = _minmax(np.array([d["sparse_score"] for d in docs], dtype=float))
        final = self.beta * ce + (1.0 - self.beta) * sp
        out = []
        for j in np.argsort(-final)[:k]:
            d = dict(docs[j])
            d.update(ce_score=float(ce[j]), final_score=float(final[j]))
            out.append(d)
        return out


if __name__ == "__main__":
    # Runnable smoke corpus (CPU). Four things this must prove, each of which fails silently:
    #   1. the two stages are split — retrieve() must not have reranked
    #   2. rerank() actually reorders, and its output is what the generator receives
    #   3. an out-of-vocabulary query yields an EMPTY pool, not k zero-score fillers
    #   4. a reloaded cache ranks identically and its score vector is exactly corpus-length
    #      (the stale-index drift: ids that index the corpus but not the document that was
    #       scored — no exception, plausible numbers, wrong MAP)
    import shutil

    corpus = [
        "The Mona Lisa is a 16th-century portrait by Leonardo da Vinci.",
        "Leonardo da Vinci was an Italian polymath of the High Renaissance.",
        "The Louvre museum is located in Paris, France.",
        "Vincent van Gogh painted The Starry Night in 1889.",
        "The Eiffel Tower is on the Champ de Mars in Paris.",
    ]
    cache = "scratch/_retriever_selfcheck"
    shutil.rmtree(cache, ignore_errors=True)

    r = Retriever(corpus, device="cpu", cache_dir=cache)
    cands = r.retrieve("Who painted the Mona Lisa?")
    assert cands and "ce_score" not in cands[0], "retrieve() must not rerank"
    res = r.rerank("Who painted the Mona Lisa?", cands, k=3)
    assert res and "Mona Lisa" in res[0]["text"], res
    assert all("ce_score" in d and "final_score" in d for d in res)
    assert r.retrieve("zzz qqq xxxx") == [], "out-of-vocabulary query must not fill the pool"

    r2 = Retriever(corpus, device="cpu", cache_dir=cache)      # must LOAD, not rebuild
    again = r2.retrieve("Who painted the Mona Lisa?")
    assert [d["doc_id"] for d in again] == [d["doc_id"] for d in cands], \
        "a reloaded index ranked differently from the one just built"
    assert all(0 <= d["doc_id"] < len(corpus) for d in again), "doc_id outside corpus range"
    assert len(r2.bm.get_scores(_tok("Paris"))) == len(corpus), \
        "score vector length != corpus length: the index and the corpus have drifted"
    shutil.rmtree(cache, ignore_errors=True)

    print("retriever OK: retrieve", [d["doc_id"] for d in cands],
          "-> rerank", [d["doc_id"] for d in res], "| cache reload identical, ids in range")
    print(f"  sparse {r.sparse_ms:.1f} ms over {len(corpus)} paragraphs (device=cpu)")
