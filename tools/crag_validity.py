"""Does the local flan-t5 PROXY evaluator agree with the vendored CRAG labels?

The real CRAG evaluator is a trained 0.77B t5-large; the benchmark uses google/flan-t5-base as
a prompt-based stand-in (`src/evaluator/crag_module.py`). Before that proxy's action distribution
is trusted anywhere, it must be scored against the only ground truth vendored in-repo:

  data/{popqa,bio,pubqa,arc_challenge}/ref/{correct,incorrect,ambiguous}

These files have no explicit question text, so questions are recovered by joining each ref
entry — normalized, first 80 alphanumeric chars of the knowledge blob — onto the test file's
"<question> [SEP] <knowledge>" lines. The folder name is the gold action
(correct->Correct, incorrect->Incorrect, ambiguous->Ambiguous).

`ponytail: agreements vary by dataset and fold: blobs labelled "incorrect" are still partially
on-topic, so a non-discriminative proxy scores them Correct. The table exposes exactly that.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from src.evaluator.crag_module import CRAGEvaluator

DATA = next(p for p in (Path("data"), Path("../Code/CRAG-main/data")) if p.exists())
DATASETS = {
    "popqa": "test_popqa.txt",
    "bio": "test_bio.txt",
    "pubqa": "test_pubqa.txt",
    "arc_challenge": "test_arc_challenge.txt",
}
EXPECTED = {"correct": "Correct", "incorrect": "Incorrect", "ambiguous": "Ambiguous"}


def _norm(s: str) -> str:
    s = re.sub(r"(?i)^\s*knowledge\d+\s*[:\-]?\s*", "", s)   # 'ambiguous' entries carry it
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:80]


def _test_index(path: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            q, _, blob = line.partition(" [SEP] ")
            if q and blob:
                idx[_norm(blob)] = q.strip()
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-class", type=int, default=30,
                    help="ref entries sampled per (dataset, folder) — CPU cost, keep small")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/crag_validity.md")
    a = ap.parse_args()

    ev = CRAGEvaluator(device=a.device)
    rows = []
    for ds in [d.strip() for d in a.datasets.split(",")]:
        idx = _test_index(DATA / ds / DATASETS[ds])
        for folder, expected in EXPECTED.items():
            entries = []
            path = DATA / ds / "ref" / folder
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    q = idx.get(_norm(line))
                    if q:
                        entries.append((q, line.strip()))
                    if len(entries) >= a.per_class:
                        break
            if not entries:
                rows.append((ds, folder, expected, 0, Counter()))
                continue
            preds = Counter()
            for q, blob in entries:
                preds[ev.evaluate(q, [blob])["action"]] += 1
            rows.append((ds, folder, expected, len(entries), preds))

    out = ["| dataset | ref folder (expected action) | match rate | proxy action distribution |",
           "|---|---|---|---|"]
    print(*out, sep="\n")
    for ds, folder, expected, n, preds in rows:
        rate = preds[expected] / n if n else 0
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(preds.items())) or "no matches"
        line = f"| {ds} | {folder} ({expected}) | {rate:.0%} | {dist} |"
        print(line)
        out.append(line)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("# CRAG proxy validity (flan-t5-base vs vendored ref labels)\n\n"
                        + "\n".join(out) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
