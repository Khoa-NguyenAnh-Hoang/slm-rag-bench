"""Static HTML report from the scorecard JSON. No server, no Streamlit.

Consumes the same `results/*/scorecard.json` schema as analysis/scorecard.py so the two
deliverables cannot drift apart. The Pareto fronts render as inline SVG — no plotting dep.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.scorecard import load_cells, cell_row, pareto, _n


def _svg_scatter(rows: list[dict], x: str, y: str, title: str, front_cells: set[str]) -> str:
    """Minimal inline SVG, faithfulness-on-y vs latency/cost-on-x. Pareto points in green."""
    xs, ys = [r[x] for r in rows if r[x] is not None and r[y] is not None], \
             [r[y] for r in rows if r[x] is not None and r[y] is not None]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs) or 1.0
    y0, y1 = min(ys), max(ys) or 1.0
    pad = 30
    W, H = 420, 240

    def P(r):
        px = pad + (r[x] - x0) / (x1 - x0 or 1) * (W - 2 * pad)
        py = H - pad - (r[y] - y0) / (y1 - y0 or 1) * (H - 2 * pad)
        return px, py

    pts = []
    for r in rows:
        if r[x] is None or r[y] is None:
            continue
        px, py = P(r)
        fill = "#1a5c2a" if r["cell"] in front_cells else "#888"
        pts.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="5" fill="{fill}">'
                   f'<title>{r["cell"]}: x={r[x]:.2f} y={r[y]:.3f}</title></circle>')
        pts.append(f'<text x="{px + 6:.0f}" y="{py + 3:.0f}" font-size="9">{r["cell"]}</text>')
    return (f'<svg width="{W}" height="{H}" style="border:1px solid #ccc">'
            f'<text x="{pad}" y="14" font-size="11">{title}</text>' + "".join(pts) + "</svg>")


def main() -> None:
    ap = argparse.ArgumentParser(description="static HTML report")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/report.html")
    a = ap.parse_args()

    rows = [cell_row(c) for c in load_cells(Path(a.results)).values()]
    f1 = {r["cell"] for r in pareto(rows, "lat_p95_ms", "faithfulness", x_min=True, y_max=True)}
    f2 = {r["cell"] for r in pareto(rows, "api_usd", "faithfulness", x_min=True, y_max=True)}
    svg1 = _svg_scatter(rows, "lat_p95_ms", "faithfulness",
                        "faithfulness vs p95 latency (green = Pareto)", f1)
    svg2 = _svg_scatter(rows, "api_usd", "faithfulness",
                        "faithfulness vs api-equivalent $ (green = Pareto)", f2)

    header = "| cell | n | adj acc % | faithfulness | lat p95 ms | api $ | gpu $ |"
    sep = "|---|---|---|---|---|---|---|"
    body = [f"| {r['cell']} | {r['n']} | {_n(r['adj_acc_%'], '.1f')} | {_n(r['faithfulness'])} | "
            f"{_n(r['lat_p95_ms'], '.0f')} | {_n(r['api_usd'], '.4f')} | {_n(r['gpu_usd'], '.4f')} |"
            for r in rows]
    table = "<pre>" + "\n".join([header, sep, *body]) + "</pre>"

    html = f"""<html><head><meta charset="utf-8"><title>slm-rag-bench scorecard</title></head>
<body style="font-family:monospace">
<h2>slm-rag-bench — scorecard</h2>
{table}
<h3>Pareto fronts</h3>
{svg1}
{svg2}
<p style="color:#888">Static report. Re-render with: <code>uv run python -m analysis.report</code>.
Green circles are Pareto-optimal cells; hover for values. No server, no live data.</p>
</body></html>"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
