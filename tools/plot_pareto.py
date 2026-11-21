"""Render the two Pareto-front figures the papers embed, as PNGs.

Reuses the scorecard layer (load_cells / cell_row / pareto) so the figures cannot
drift from the numbers in results/scorecard_summary.md and the HTML report — the
same fronts, computed once, rendered three ways.

    uv run --with matplotlib python -m tools.plot_pareto

Writes <out-dir>/pareto_latency.png (faithfulness vs P95 latency) and
pareto_cost.png (faithfulness vs GPU-amortised cost). Green = on the front.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.scorecard import cell_row, load_cells, pareto


def _scatter(rows, x, y, x_label, title, out_path):
    import matplotlib

    matplotlib.use("Agg")           # no display on a headless box
    import matplotlib.pyplot as plt

    front = {r["cell"] for r in pareto(rows, x, y, x_min=True, y_max=True)}
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    for r in rows:
        if r[x] is None or r[y] is None:
            continue
        on_front = r["cell"] in front
        ax.scatter(r[x], r[y], s=42,
                   color="#1a5c2a" if on_front else "#888888", zorder=3)
        ax.annotate(r["cell"], (r[x], r[y]), textcoords="offset points",
                    xytext=(6, 5), fontsize=6.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("ragas faithfulness")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path} (front: {sorted(front)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="course_paper/figures")
    args = ap.parse_args()

    rows = [cell_row(c) for c in load_cells(Path(args.results)).values()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _scatter(rows, "lat_p95_ms", "faithfulness", "P95 latency (ms)",
             "Faithfulness vs P95 latency (n=30, green = Pareto)",
             out_dir / "pareto_latency.png")
    _scatter(rows, "gpu_usd", "faithfulness", "GPU-amortised cost per query (USD)",
             "Faithfulness vs GPU cost (n=30, green = Pareto)",
             out_dir / "pareto_cost.png")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
