"""50-sample human-agreement harness.

Why it exists: the faithfulness axis is "Correct/Hallucinated/NoAnswer/Unknown" from
deterministic substring labels, and judge-scored faithfulness comes from an LLM. Neither is
defensible without a human checking a stratified slice. This file makes that slice cheap to
produce and the agreement number reproducible.

Usage:
  python -m src.monitor.labels export [--results results] [--n 50] [--out results/labels_export.csv]
  python -m src.monitor.labels score  results/labels_export.csv   # after filling human_label

Labeling: for each row, read question + contexts + the model's answer, then judge with Mala's
3-label rubric (Factual / Hallucination / Refusal), from
Code/Beyond_Retrieval-main/04_evaluation/llm_as_judge.ipynb — the SAME language the LLM judge
uses, so the agreement table means "human vs the judge's notion of the same three categories".
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("Factual", "Hallucination", "Refusal")
DET_TO_HUMAN = {"Correct": "Factual", "Hallucinated": "Hallucination", "NoAnswer": "Refusal"}


def _rows(results_root: Path) -> list[dict]:
    out = []
    for card_path in sorted(results_root.glob("*/scorecard.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        cell = card.get("cell") or card_path.parent.name
        raf = (card.get("ragas") or {}).get("faithfulness")
        for q in card.get("queries", []):
            out.append({
                "cell": cell,
                "query_id": q.get("extra", {}).get("query_id", q.get("query_id",
                           q.get("question_id", ""))),
                "question": q.get("question", ""),
                "answer": (q.get("answer") or "").replace("\n", " "),
                "contexts": " ||| ".join(q.get("contexts") or [])[:1500],
                "action": q.get("action") or "",
                "det_label": (q.get("extra") or {}).get("label") or q.get("label", ""),
                "ragas_faithfulness": raf if raf is not None else "",
            })
    return out


def export(results_root: Path, n: int, out: Path, seed: int = 42) -> None:
    rows = _rows(results_root)
    if not rows:
        raise SystemExit(f"no scorecards under {results_root} — run at least one cell first")
    # Round-robin across (cell, det_label) buckets so every cell and every label is represented
    # as evenly as 50 rows over 6 cells will allow. A purely random sample commonly drops a
    # rare bucket's only cell entirely.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["cell"], r["det_label"])].append(r)
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    pool, picked = sorted(buckets.values(), key=len, reverse=True), []
    while len(picked) < n and any(pool):
        for b in pool:
            if b and len(picked) < n:
                picked.append(b.pop())
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cell", "query_id", "question", "contexts", "answer", "action",
              "det_label", "ragas_faithfulness", "human_label", "notes"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in picked:
            w.writerow({**{k: r[k] for k in fields[:-2]}, "human_label": "", "notes": ""})
    per_label = Counter(r["det_label"] for r in picked)
    print(f"wrote {len(picked)} rows -> {out}")
    print("det_label mix:", dict(per_label))
    print(f"cells: {len({r['cell'] for r in picked})}, seed={seed}")


def _kappa(human: list[str], machine: list[str]) -> float:
    """Cohen's kappa over the shared label set."""
    n = len(human)
    if n == 0:
        return float("nan")
    observed = sum(h == m for h, m in zip(human, machine)) / n
    labels = sorted(set(human) | set(machine))
    chance = sum((human.count(l) / n) * (machine.count(l) / n) for l in labels)
    return (observed - chance) / (1 - chance) if chance < 1 else 1.0


def score(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    pairs = []
    for r in rows:
        human = (r["human_label"] or "").strip()
        if not human:
            continue
        mapped = DET_TO_HUMAN.get(r["det_label"].strip())
        if mapped:
            pairs.append((human, mapped))
    if not pairs:
        raise SystemExit("no rows with a human_label yet — fill the CSV first")
    n = len(pairs)
    human = [p[0] for p in pairs]
    machine = [p[1] for p in pairs]
    print(f"n scored: {n} / {len(rows)} exported")
    print(f"agreement: {sum(h == m for h, m in pairs) / n:.1%}")
    print(f"kappa:      {_kappa(human, machine):.2f}")
    # Confusion in the format the report embeds verbatim.
    confusion: dict[str, Counter] = defaultdict(Counter)
    for h, m in pairs:
        confusion[h][m] += 1
    sets = list(LABELS) + sorted({m for row in confusion.values() for m in row}
                                 - set(LABELS))
    print("\nrows = human, cols = deterministic label mapped to the human rubric:")
    print("| | " + " | ".join(sets) + " |")
    print("|---" * (len(sets) + 1) + "|")
    for h in LABELS:
        print("| " + h + " | " + " | ".join(str(confusion[h][m]) for m in sets) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--results", default="results")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--out", default="results/labels_export.csv")
    e.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("path", nargs="?", default="results/labels_export.csv")
    a = ap.parse_args()
    if a.cmd == "export":
        export(Path(a.results), a.n, Path(a.out), a.seed)
    else:
        score(Path(a.path))


if __name__ == "__main__":
    main()
