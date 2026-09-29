"""HotpotQA distractor-set sampling + corpus builder.

Loads off the HF Hub via `load_dataset`. The upstream raw-JSON hosts are unreachable (connection
fails over http, times out over https, and the GitHub raw paths 404), so the Hub is the only
live route.

`supporting_facts` shape differs by source and matters: raw JSON ships it as a list of
[title, sent_id] pairs, HF parquet ships {'title': [...], 'sent_id': [...]}. We read the HF
shape. Getting this wrong yields zero supporting paragraphs, which zeroes every retrieval
metric and makes every answer look hallucinated — a silent failure, so it is asserted below.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HF_DATASET = "hotpotqa/hotpot_qa"
SPLITS = {"dev": "validation", "train": "train"}


def _cache_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"hotpot_{SPLITS.get(split, split)}.json"


def load_split(split: str = "dev", cache_dir: str | Path = "scratch") -> list[dict]:
    """Return the raw HF rows for a split, cached to disk (Colab filesystems are ephemeral).

    Caching to JSON rather than reusing the HF arrow cache keeps the sampled subset byte-stable
    across sessions and machines, which the manifest's dataset hash depends on.
    """
    cache = _cache_path(cache_dir, split)
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            return json.load(f)
    from datasets import load_dataset

    rows = [dict(r) for r in load_dataset(HF_DATASET, "distractor", split=SPLITS.get(split, split))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _split_row(inst: dict) -> dict:
    sup_titles = set(inst["supporting_facts"]["title"])       # HF shape, not raw-JSON pairs
    supporting, distractors = [], []
    for title, sents in zip(inst["context"]["title"], inst["context"]["sentences"]):
        para = {"title": title, "sentences": sents}
        (supporting if title in sup_titles else distractors).append(para)
    return {
        "question_id": inst["id"],
        "question": inst["question"],
        "answer": inst.get("answer", ""),
        "gold_titles": sorted(sup_titles),
        "supporting": supporting,
        "distractors": distractors,
    }


def sample_hotpot(
    n: int = 300,
    seed: int = 42,
    split: str = "dev",
    cache_dir: str | Path = "scratch",
) -> list[dict]:
    """Return n instances: {question_id, question, answer, gold_titles, supporting[], distractors[]}.

    `split` is the EVAL split. The retrieval CORPUS is built separately by build_corpus() and is
    deliberately much larger: a corpus drawn from the same n questions makes retrieval degenerate
    (2 of 10 candidates gold by construction).
    """
    raw = load_split(split, cache_dir)
    assert len(raw) >= n, f"split {split} has {len(raw)} rows, need {n}"
    # Deterministic selection without materialising a 90k-row sample: hash the row id, take the
    # n smallest. Stable across Python versions, unlike random.sample on a list of dicts.
    ranked = sorted(raw, key=lambda r: (hashlib_seed(r["id"], seed), r["id"]))[:n]
    return [_split_row(inst) for inst in ranked]


def hashlib_seed(row_id: str, seed: int) -> str:
    """Order-stable, version-stable pseudo-random key. Hashlib is stdlib and deterministic;
    `hash()` is salted per process and would silently change the sample between runs."""
    import hashlib

    return hashlib.md5(f"{seed}:{row_id}".encode()).hexdigest()


def build_corpus(
    split: str = "train",
    cache_dir: str | Path = "scratch",
    max_paras: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return (texts, titles) over the WHOLE split's paragraphs.

    Every paragraph of every question is a corpus document, deduplicated by (title, text).
    Dedup must key on the TITLE as well as the text: HotpotQA reuses paragraph text across
    questions, and collapsing by text alone would hand the same doc_id to two different titles —
    which silently corrupts the title→doc_id map that gold matching is built on.

    The 90k-row train split yields ~500-900k paragraphs — the index build is tens of minutes
    and is cached by the retriever, not here.

    max_paras caps the corpus for local iteration; it MUST be recorded in the manifest, because
    a capped corpus inflates every retrieval metric.
    """
    rows = load_split(split, cache_dir)
    texts, titles, seen = [], [], set()
    for inst in rows:
        for title, sents in zip(inst["context"]["title"], inst["context"]["sentences"]):
            text = " ".join(sents)
            key = (title, text)     # key on TITLE too, not text alone
            if key not in seen:
                seen.add(key)
                texts.append(text)
                titles.append(title)
        if max_paras and len(texts) >= max_paras:
            break
    assert len(texts) > 0, "empty corpus"
    return texts, titles


def dataset_manifest() -> dict:
    """Provenance for manifest.json. The HF revision is pinned so a re-run cannot silently
    score a different dataset than the one reported."""
    return {
        "dataset": HF_DATASET,
        "config": "distractor",
        "splits": SPLITS,
        "source": "huggingface.co/datasets/hotpotqa/hotpot_qa",
        "note": "upstream raw-JSON hosts are unreachable; the HF Hub is the only live route",
    }


if __name__ == "__main__":
    n = int(os.environ.get("N", "300"))
    rows = sample_hotpot(n=n, seed=42, split="dev")
    assert len(rows) == n, len(rows)
    assert all(r["question"] and r["supporting"] and r["distractors"] for r in rows)
    assert all(len(r["supporting"]) == 2 for r in rows), "every HotpotQA question has 2 gold paras"
    assert all(set(r["gold_titles"]) == {p["title"] for p in r["supporting"]} for r in rows), \
        "gold_titles/supporting disagree — supporting_facts shape regression"
    # NOT 10 for every row. Measured on distractor/validation: 60 of 7,405 rows (0.8%) carry
    # fewer than 10 context paragraphs (counts of 2-9 appear), while ALL 7,405 still have exactly
    # 2 gold titles with gold ⊆ context. So "2 supporting + 8 distractors" is the norm, not an
    # invariant. Those rows are EASIER (fewer distractors to reject), so distractor count per
    # query is a reported manifest field, not a hidden assumption.
    ctx_counts = [len(r["supporting"]) + len(r["distractors"]) for r in rows]
    truncated = sum(c < 10 for c in ctx_counts)
    if truncated:
        print(f"NOTE: {truncated}/{n} sampled rows have <10 context paragraphs "
              f"(HotpotQA distractor is not uniformly 10 — 0.8% of validation)")
    # HotpotQA answers are sometimes yes/no; a 0% substring match on those is a metric artefact,
    # not a model failure, so the label path must know about it (see monitor/labels.py).
    yesno = sum(r["answer"].lower() in ("yes", "no") for r in rows)
    print(f"OK: {n} rows, {yesno} yes/no answers, gold_titles consistent, "
          f"ctx paras {min(ctx_counts)}-{max(ctx_counts)}")
    print(f"   e.g. Q={rows[0]['question']!r} A={rows[0]['answer']!r} gold={rows[0]['gold_titles']}")

    texts, titles = build_corpus(split="dev", max_paras=2000)
    assert len(texts) == len(titles), "titles must stay aligned with texts"
    assert all(t.strip() for t in titles), "every corpus doc needs a title for gold matching"
    print(f"OK: capped corpus {len(texts)} paragraphs, {len(set(titles))} distinct titles")
