# The fixed retriever: why it is bm25s + MiniLM, and nothing else

The benchmark's question is **which small model should I pick**. That only has an answer if
retrieval is held constant, so the retriever is fixed and every cell uses the same one. This
file records the measurements that chose it, so the choice is not a preference someone has to
re-litigate later.

**Shipped:** `Retriever` in `retriever.py` — `bm25s` (lucene scoring) for a depth-30 candidate
pool, `cross-encoder/ms-marco-MiniLM-L-6-v2` to cut it to 10. 91 MB of weights, no JDK, no
FAISS, no embedding build.

## The decision, in one number

The dense arm was removed because its headroom was arithmetically capped, not because it lost
a close comparison.

At the full corpus, sparse-only **recall@30 is 0.9800**. The candidate pool is 30 deep, so gold
is already in the pool 98% of the time. After reranking, **recall@10 is 0.9767**. A dense arm
fused into that same pool could therefore raise recall@10 by **at most 0.0033** — up to the
0.9800 that sparse-only already reaches at depth 30. There is no configuration of fusion and
reranking that beats that ceiling, because the ceiling is the pool.

In exchange it cost a 438 MB model, a `faiss` dependency, and tens of minutes of encoding
before the first query. It also put retrieval variance into every cell, which is the one thing
a fixed retriever exists to prevent.

## The measurements

**Full scale** — whole train split, **483,696** paragraphs after (title, text) dedup, n=300
eval subset at seed 42, 0 queries with gold absent from the index, CPU:

| k | 1 | 3 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|
| recall@k, sparse only | 0.7133 | 0.8467 | 0.8900 | 0.9400 | **0.9800** | 0.9867 |

After reranking the top-30 pool down to 10:

| metric | sparse only @10 | after MiniLM rerank |
|---|---|---|
| recall@10 | 0.9400 | **0.9767** |
| recall@3 | 0.8467 | **0.8800** |
| map@3 | — | **0.8417** |
| ndcg@10 | — | **0.9115** |

So the reranker is what earns its place (+0.0367 recall@10 on top of a pool that is already
98% correct at depth 30), and the dense arm is what does not.

**Arm comparison** — 3,944 real paragraphs from a contiguous block of 400 questions, 150 eval
queries drawn from inside that block, CPU:

| arm | R@10 | R@3 | MAP@3 | nDCG@10 | ms/q |
|---|---|---|---|---|---|
| bm25s → MiniLM | 0.9933 | 0.8667 | **0.8167** | 0.9147 | 1618 |
| bm25s only | 0.9667 | 0.6567 | 0.6056 | 0.7826 | 0.4 |
| bm25s + MPNet fusion | 0.9667 | 0.8000 | 0.7300 | 0.8551 | 58 |
| fusion → MiniLM | 0.9933 | 0.8600 | 0.8078 | 0.9113 | 1785 |

Sparse+rerank matches the full hybrid+rerank arm and beats fusion-without-rerank. R@10 is
saturated at this scale and so cannot separate anything — the R@3 and MAP@3 columns are where
the arms actually differ, and the ordering there is the same one the full-scale ceiling
argument reaches independently.

Two traps in running that comparison, both of which produced a meaningless table before they
were fixed:

- `build_corpus(max_paras=N)` walks rows in **file order** and stops at N. A hash-sampled eval
  query's gold is almost never inside a capped corpus — measured **2 of 200** queries with
  gold, at which point every arm scores 0.5 and the comparison is noise. A capped corpus must
  be built from a block that *contains* the eval queries, which is what locked decision 6
  requires anyway.
- The corpus is **483,696** paragraphs, not the ~904k that code comments and the plan claimed.
  The real number matters: it is the `depth` every retrieval metric is reported at.

The 1618 ms/q for the rerank is a **CPU** number — 30 pairs through a 6-layer cross-encoder on
4 torch threads. On a T4 it is a few ms against 3–6 s of generation. Unmeasured on GPU.

## Why bm25s, and why `method="lucene"`

`bm25s` replaced `pyserini` (needed JDK 17, and could not be measured on a machine without it)
and `rank_bm25` (249.72 ms/query at 50k paragraphs, which extrapolates to seconds at corpus
scale and would have dominated end-to-end latency entirely). Measured at 50,002 paragraphs:
bm25s 1.85 ms/query, 4.4 s build; `rank_bm25` 249.72 ms/query, 2.4 s build. At 483,696
paragraphs bm25s is 20.8 ms/query with a 42.2 s build. The index is portable `.npy`/`.json`,
not a JVM-versioned binary directory.

`method="lucene"` is not the wheel default and the reason is a silent failure, not a
ranking difference. On a Zipf sweep, `lucene` and `atire` produce **identical** top-10 on every
query (Jaccard 1.000). `robertson` differs (Jaccard 0.946) because it **floors IDF at zero**:
a query of only high-df terms then scores exactly 0.0 against all 20,000 documents, the `> 0`
filter empties the pool, and a run that believes it is retrieving returns nothing. It buys
nothing over `lucene` and carries that landmine.

## Three integration traps, all silent, all now covered by asserts

1. **`load()` is a `@classmethod` that RETURNS a new object.** `bm = BM25(); bm.load(dir)`
   leaves `bm.scores` unset and the next query dies with `AttributeError`. Must be
   `bm = BM25.load(dir)`.
2. **The wheel's `tokenize()` defaults diverge from the repo's.** It defaults to
   `stopwords='english'` and its own `token_pattern`, so index and query can tokenise
   differently. Worse, an all-stopword query tokenises to an empty list and the backend
   returns **k documents at score 0.0** instead of raising. Fixed by building the index from
   explicit `_tok()` token lists, so tokenisation is one shared decision and not a per-backend
   default.
3. **Out-of-vocabulary terms score 0.0 against every document, silently.** HotpotQA questions
   are full of proper nouns. The `> 0` filter is what stops a query with no lexical overlap
   from filling the pool with arbitrary documents that then get reranked as if they were
   candidates. `retriever.py`'s self-check asserts an OOV query returns an empty pool.

All three are encoded in `retriever.py` and its self-check, not left as prose here.

## What this does not establish

- **No GPU measurement of the reranker's latency.** 1618 ms/q is CPU. The cell-level latency
  numbers will come from the first real run.
- **No comparison against `pyserini` on this corpus.** There is no JDK on this machine, so
  "bm25s is Nx faster than pyserini" is not a number that exists. Do not quote one.
- **The 3,944-paragraph table's absolute values do not extrapolate.** R@10 saturates there.
  Only the full-scale table is quotable, and only for the fixed retriever at 483,696
  paragraphs, seed 42, n=300.
- **This was decided before any cell ran, deliberately.** Swapping the retriever mid-grid
  would change every retrieval metric and make cells incomparable. It is recorded as a
  deviation in every `manifest.json`.
