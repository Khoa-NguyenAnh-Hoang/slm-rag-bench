"""Prefetch everything run_cell needs, on cheap CPU/net — not on paid GPU clock.

Run once per fresh box BEFORE renting time matters:

    uv run python -m tools.prefetch

Pulls: HotpotQA splits (at the pinned HF_REVISION), tokenizers for every
serve_id, MiniLM reranker, flan-t5-base proxy, MPNet embeddings, and all
four vLLM checkpoints (3 generators + judge). No GPU, no server start.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    n = 0

    from src.data.hotpot_loader import load_split

    for split in {cfg["eval_split"], cfg["corpus"]["split"]}:
        rows = load_split(split)
        print(f"dataset {split}: {len(rows)} rows")
        n += 1

    serve_ids = [m["serve_id"] for m in cfg["models"].values()] + [cfg["judge"]["serve_id"]]
    for sid in dict.fromkeys(serve_ids):
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(sid)
        print(f"tokenizer OK: {sid}")
        n += 1
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=sid)
            print(f"weights OK: {sid}")
        except Exception as e:
            print(f"weights SKIP {sid}: {type(e).__name__}: {e} (vLLM will fetch at serve)")
        n += 1

    from sentence_transformers import CrossEncoder

    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    print("reranker OK")
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    T5Tokenizer.from_pretrained(cfg["crag"]["model"])
    T5ForConditionalGeneration.from_pretrained(cfg["crag"]["model"])
    print(f"crag OK: {cfg['crag']['model']}")
    from ragas.embeddings import HuggingFaceEmbeddings

    HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2", device="cpu")
    print("embeddings OK")
    print(f"prefetch done: {n} items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
