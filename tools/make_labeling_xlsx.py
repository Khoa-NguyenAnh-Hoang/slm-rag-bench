"""Labeling workbook: CSV -> .xlsx with a label dropdown, and back.

The human-agreement deliverable needs the human to label WITHOUT seeing the
deterministic labels they are measured against. This tool splits the export:
the "labeling" sheet carries everything to judge from plus a dropdown limited
to the three rubric values; det_label / ragas_faithfulness live on a separate
"reference" sheet. `collect` merges the filled labels back into the exact CSV
schema `src.monitor.labels score` expects.

    uv run --with openpyxl python -m tools.make_labeling_xlsx build     # csv -> xlsx
    uv run --with openpyxl python -m tools.make_labeling_xlsx collect   # filled xlsx -> csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CSV_PATH = Path("results/labels_export.csv")
XLSX_PATH = Path("results/labels_workbook.xlsx")
LABELS = ("Factual", "Hallucination", "Refusal")
LABEL_COLS = ["cell", "query_id", "question", "contexts", "answer", "action",
              "human_label", "notes"]
REF_COLS = ["cell", "query_id", "det_label", "ragas_faithfulness"]
ALL_COLS = ["cell", "query_id", "question", "contexts", "answer", "action",
            "det_label", "ragas_faithfulness", "human_label", "notes"]
WIDTHS = {"cell": 20, "query_id": 16, "question": 40, "contexts": 80,
          "answer": 40, "action": 10, "human_label": 16, "notes": 24,
          "det_label": 14, "ragas_faithfulness": 16}


def _set_widths(ws, cols):
    from openpyxl.utils import get_column_letter

    for idx, name in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = WIDTHS[name]


def build() -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8", newline="")))
    wb = Workbook()
    ws = wb.active
    ws.title = "labeling"
    ws.append(LABEL_COLS)
    for r in rows:
        ws.append([r[c] for c in LABEL_COLS])
    for c in ws[1]:
        c.font = Font(bold=True)
    _set_widths(ws, LABEL_COLS)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"',
                        allow_blank=True, showErrorMessage=True,
                        errorTitle="Rubric labels only",
                        error="Use Factual, Hallucination or Refusal.")
    ws.add_data_validation(dv)
    dv.add(f"H2:H{len(rows) + 1}")
    ws.freeze_panes = "A2"

    ref = wb.create_sheet("reference")
    ref.append(REF_COLS)
    for r in rows:
        ref.append([r[c] for c in REF_COLS])
    for c in ref[1]:
        c.font = Font(bold=True)
    _set_widths(ref, REF_COLS)

    wb.save(XLSX_PATH)
    print(f"wrote {XLSX_PATH}: {len(rows)} rows; label on the 'labeling' sheet "
          f"(dropdown), det_label/ragas on 'reference'")
    return 0


def collect() -> int:
    from openpyxl import load_workbook

    wb = load_workbook(XLSX_PATH)
    ws, ref = wb["labeling"], wb["reference"]
    header = [c.value for c in ws[1]]
    assert header == LABEL_COLS, header
    lookup = {}
    for row in ref.iter_rows(min_row=2, values_only=True):
        lookup[(row[0], str(row[1]))] = dict(zip(REF_COLS, row))
    out, bad, filled = [], [], 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        r = dict(zip(LABEL_COLS, row))
        if r["human_label"] is not None:
            r["human_label"] = str(r["human_label"]).strip()
        if r["human_label"] and r["human_label"] not in LABELS:
            bad.append((r["cell"], r["query_id"], r["human_label"]))
        filled += bool(r["human_label"])
        merged = {c: "" for c in ALL_COLS}
        merged.update({c: ("" if v is None else str(v)) for c, v in r.items()})
        merged["human_label"] = r["human_label"] or ""
        key = (r["cell"], str(r["query_id"]))
        if key in lookup:
            merged["det_label"] = str(lookup[key]["det_label"] or "")
            merged["ragas_faithfulness"] = str(lookup[key]["ragas_faithfulness"] or "")
        out.append(merged)
    if bad:
        for cell, qid, val in bad:
            print(f"BAD LABEL {cell}/{qid}: {val!r} — allowed: {LABELS}", file=sys.stderr)
        return 1
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {CSV_PATH}: {filled}/50 rows labeled "
          + ("" if filled == len(out) else f"({len(out) - filled} left blank — "
                                            "score() skips blanks)"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("build", "collect"))
    a = ap.parse_args()
    return build() if a.cmd == "build" else collect()


if __name__ == "__main__":
    sys.exit(main())
