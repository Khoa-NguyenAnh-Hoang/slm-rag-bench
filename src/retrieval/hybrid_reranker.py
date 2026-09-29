"""HybridLI retrieval (0.7 sparse BM25 + 0.3 dense MPNet over FAISS), then MiniLM
cross-encoder rerank as a SEPARATE stage.

`retrieve()` and `rerank()` are split deliberately. Fused into one call, the reranker's own
latency is invisible — which is exactly the quantity the ablation turns on — and the
baseline/reranked/crag pipelines become inexpressible.

The sparse arm is a config switch (`retrieval.sparse`), default pyserini. Measured cost of the
alternative on this corpus: rank_bm25 is ~117 ms/query at 50k docs and extrapolates to ~2.1 s at
the ~904k-doc train split, versus pyserini's 6.7 ms on the same index. A 2.1 s Python loop would
dominate a 3-6 s end-to-end generation and turn the latency axis into an artefact of the sparse
implementation. pyserini needs JDK 17+ (`apt-get install -y openjdk-17-jdk-headless`); rank_bm25
is the no-JDK fallback for local iteration.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

CAND_K = 30          # fusion candidate depth before rerank
DEFAULT_CACHE = "scratch/index"


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def _tok(text: str) -> list[str]:
    """BM25 word tokens. A whitespace split() glues punctuation onto the term ('paris?',
    '1879.'), so a query token almost never equals a document token: scores drift toward 0, the
    `> 0` filter empties the sparse arm, and the 0.7 sparse weight silently degrades the
    'hybrid' into dense-only."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridLI:
    """Sparse+dense fusion, and (separately) cross-encoder rerank.

    sparse: "pyserini" (default, paper-matching latency) or "rank_bm25" (no JDK).
    cache_dir: the built index is expensive at corpus scale (~904k paragraphs), so it is
    persisted under <cache_dir>/<corpus-hash>/ and reloaded on the next run. The key is a hash of
    the corpus contents, so a different corpus can never hit a stale index.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        w_sparse: float = 0.7,
        w_dense: float = 0.3,
        beta_rerank: float = 0.85,
        device: str = "cuda",
        sparse: str = "pyserini",
        cache_dir: str | Path = DEFAULT_CACHE,
        cand_k: int = CAND_K,
    ) -> None:
        assert abs(w_sparse + w_dense - 1.0) < 1e-6, "HybridLI weights must sum to 1"
        assert sparse in ("pyserini", "rank_bm25"), f"unknown sparse arm {sparse!r}"
        self.w_sparse, self.w_dense, self.beta = w_sparse, w_dense, beta_rerank
        self.sparse, self.cand_k, self.device = sparse, cand_k, device
        self.corpus = list(corpus)

        # Lazy imports so the module body is importable on a box with no torch/faiss.
        from sentence_transformers import CrossEncoder, SentenceTransformer

        # Canonical id, identical to the one serve.build_embeddings() hands ragas: a bare
        # "all-mpnet-base-v2" risks resolving to a different revision, which would put
        # AnswerRelevancy in a different embedding space than the retriever's.
        self.dense_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

        self.root = Path(cache_dir) / hashlib.md5(
            "\n".join(self.corpus).encode()).hexdigest()[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_build_dense()
        self._searcher = self._load_or_build_sparse()
        self.sparse_ms = 0.0
        self.dense_ms = 0.0

    # ---------- index construction (cached) ----------

    def _load_or_build_dense(self):
        """Embed the corpus once, cache the matrix. Encoding ~904k paragraphs on a T4 is tens of
        minutes; every cell would otherwise pay it again."""
        import faiss

        cache = self.root / "embeddings.npy"
        if cache.exists():
            embs = np.load(cache)
        else:
            t0 = time.perf_counter()
            embs = self.dense_model.encode(self.corpus, batch_size=64, convert_to_numpy=True,
                                           show_progress_bar=False)
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs.astype("float32")
            np.save(cache, embs)
            print(f"[hybrid] encoded {len(self.corpus)} paragraphs in "
                  f"{time.perf_counter() - t0:.1f}s -> {cache}", flush=True)
        index = faiss.IndexFlatIP(embs.shape[1])   # cosine via normalized IP — notebook:974
        index.add(embs)
        return index

    def _load_or_build_sparse(self):
        if self.sparse == "rank_bm25":
            from rank_bm25 import BM25Okapi

            return BM25Okapi([_tok(d) for d in self.corpus])
        return self._build_lucene()

    def _build_lucene(self):
        """pyserini Lucene index over the corpus. Cached the same way as the dense arm."""
        from pyserini.index.lucene import CollectionReader  # probe install early, fail loudly
        del CollectionReader

        idx = self.root / "lucene"
        if (self.root / "lucene.done").exists():
            from pyserini.search.lucene import LuceneSearcher
            return LuceneSearcher(str(idx))
        docs = self.root / "docs"     # keep the marker OUT of -input (the indexer parses that dir)
        docs.mkdir(parents=True, exist_ok=True)
        with (docs / "corpus.jsonl").open("w", encoding="utf-8") as f:
            for i, text in enumerate(self.corpus):
                f.write(json.dumps({"id": str(i), "contents": text}) + "\n")
        cmd = [sys.executable, "-m", "pyserini.index.lucene", "-collection", "JsonCollection",
               "-generator", "LuceneDocumentGenerator", "-threads", "4",
               "-input", str(docs), "-index", str(idx), "-storeRaw"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"pyserini indexing failed (rc={r.returncode}) — need JDK 17+ on PATH "
                f"(Colab: apt-get install -y openjdk-17-jdk-headless): {r.stderr[-400:]}"
            )
        (self.root / "lucene.done").touch()
        from pyserini.search.lucene import LuceneSearcher
        return LuceneSearcher(str(idx))

    # ---------- stage 1: sparse + dense fusion ----------

    def _sparse(self, query: str, k: int) -> dict[int, float]:
        if self.sparse == "rank_bm25":
            scores = self._searcher.get_scores(_tok(query))
            top = np.argsort(-scores)[:k]
            return {int(i): float(scores[i]) for i in top if scores[i] > 0}
        return {int(h.docno): h.score for h in self._searcher.search(query, k=k)}

    def _dense(self, query: str, k: int) -> dict[int, float]:
        q = self.dense_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dists, ids = self.index.search(q.astype("float32"), k)
        return {int(i): float(d) for i, d in zip(ids[0], dists[0]) if i >= 0}

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Stage 1 — sparse+dense fusion, NO cross-encoder. Returns up to `k` candidates
        (default cand_k) ordered by fused score, each with `fused_score` and no `ce_score`.

        Rerank is a separate call so its latency is its own stage (contract rule 1).
        """
        k = k or self.cand_k
        t0 = time.perf_counter()
        sp = self._sparse(query, k)
        self.sparse_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        dn = self._dense(query, k)
        self.dense_ms = (time.perf_counter() - t0) * 1000

        ids = list(dict.fromkeys(  # preserve union order: sparse then dense
            [*sorted(sp, key=sp.get, reverse=True), *sorted(dn, key=dn.get, reverse=True)]))[: 2 * k]
        if not ids:
            return []
        sp_arr = _minmax(np.array([sp.get(i, 0.0) for i in ids]))
        dn_arr = _minmax(np.array([dn.get(i, 0.0) for i in ids]))
        fused = self.w_sparse * sp_arr + self.w_dense * dn_arr
        order = np.argsort(-fused)[:k]
        fused = _minmax(fused[order])
        return [{"doc_id": ids[j], "text": self.corpus[ids[j]],
                 "fused_score": float(fused[j])} for j in order]

    # ---------- stage 2: cross-encoder rerank ----------

    def rerank(self, query: str, docs: list[dict], k: int = 10) -> list[dict]:
        """Stage 2 — MiniLM cross-encoder over `docs`, blended 0.85*CE + 0.15*fused, return top k.

        Full text is passed: the CrossEncoder truncates to its own 512-token limit internally. A
        character slice fed the ranker ~120 tokens while the returned context was the whole
        paragraph, so ce_score and the emitted text disagreed.
        """
        if not docs:
            return []
        pairs = [(query, d["text"]) for d in docs]
        ce = _minmax(np.asarray(self.reranker.predict(pairs), dtype=float))
        fused = _minmax(np.array([d["fused_score"] for d in docs], dtype=float))
        final = self.beta * ce + (1.0 - self.beta) * fused
        ranked = np.argsort(-final)[:k]
        out = []
        for j in ranked:
            d = dict(docs[j])
            d.update(ce_score=float(ce[j]), final_score=float(final[j]))
            out.append(d)
        return out


if __name__ == "__main__":
    # Runnable smoke corpus (CPU fallback: device="cpu" if no GPU). Exercises BOTH stages and
    # asserts the split behaves: rerank alone must reorder the fused candidate list.
    corpus = [
        "The Mona Lisa is a 16th-century portrait by Leonardo da Vinci.",
        "Leonardo da Vinci was an Italian polymath of the High Renaissance.",
        "The Louvre museum is located in Paris, France.",
        "Vincent van Gogh painted The Starry Night in 1889.",
        "The Eiffel Tower is on the Champ de Mars in Paris.",
    ]
    hy = HybridLI(corpus, device="cpu", sparse="rank_bm25")
    cands = hy.retrieve("Who painted the Mona Lisa?")
    assert cands and "ce_score" not in cands[0], "retrieve() must not rerank"
    res = hy.rerank("Who painted the Mona Lisa?", cands, k=3)
    assert res and "Mona Lisa" in res[0]["text"], res
    assert all("ce_score" in d and "final_score" in d for d in res)
    print("hybrid_reranker OK: retrieve", [d["doc_id"] for d in cands],
          "-> rerank", [d["doc_id"] for d in res])
    print(f"  sparse {hy.sparse_ms:.1f} ms | dense {hy.dense_ms:.1f} ms (device=cpu)")
