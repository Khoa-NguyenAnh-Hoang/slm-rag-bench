"""Run the whole grid, one cell at a time, and survive the box dying.

    uv run python -m src.runner.matrix                    # every cell
    uv run python -m src.runner.matrix --only crag        # one pipeline, all 3 models
    uv run python -m src.runner.matrix --n 30             # pilot the grid
    uv run python -m src.runner.matrix --skip-finished    # resume after a crash

WHY THIS EXISTS. `run_cell` takes a single `--only <cell>`, so firing the grid unattended meant
a shell loop written from memory — and the config advertised `python -m src.runner.matrix` as
the "run everything" command for a module that did not exist. On an interactive box you notice
that in 30 seconds. On a rented GPU you are not there, so you notice it at hour 8.

ONE CELL PER SUBPROCESS, ON PURPOSE. `run_cell` starts vLLM as a managed subprocess and kills it
on exit; the GPU memory it holds does not come back reliably inside a long-lived parent. A fresh
process per cell is the difference between six cells and four cells plus two mysterious CUDA
errors. The cost is a vLLM cold start per cell (~2-4 min), which is the same cost either way
because that is what `run_cell` already pays.

EXIT CODES ARE THE POINT. 0 = every requested cell passed its gates. 1 = at least one cell
failed a gate. 2 = nothing ran (bad config, or every requested cell was already finished).
Any other cell's failure is reported per-cell and the loop CONTINUES, because a grid that
aborts on the first failure gives you one number instead of six, and the failing cell is
usually the interesting one. Pass `--stop-on-fail` when you want the opposite.

RESUMING. `--skip-finished` treats a cell as done if its scorecard exists AND all gates in it
passed, so a crashed run resumes at the first cell that did not finish cleanly. Re-running a
cell is otherwise safe: `run_cell` unlinks the cell's `queries.jsonl` on entry.

FOR UNATTENDED USE (this is the rented-GPU path):

    nohup python -m src.runner.matrix --skip-finished > matrix.log 2>&1 &

`run_cell` writes `manifest.json` before it loads any model, so a cell that died mid-run leaves
an attributable artifact rather than nothing. Every cell's `scorecard.json` and `queries.jsonl`
land under `results/<cell_id>/`, which is NOT gitignored — commit them as they land, because a
reclaimed box takes the filesystem with it and git is the only copy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.pipeline import PIPELINES
from src.runner.run_cell import MODEL_KEYS, cell_id


def all_cells(cfg: dict, only_pipeline: str | None) -> list[str]:
    """Every cell id, cheapest first.

    Order is 1.5B -> 4B -> 8B within each pipeline, and baseline before crag. Two reasons: a
    gate or environment fault shows up on the cheapest model and the smallest pipeline first,
    where it costs two minutes instead of two hours; and if the box is reclaimed mid-grid the
    cells that did land are the ones that make the headline cost/latency table.
    """
    if only_pipeline and only_pipeline not in PIPELINES:
        raise SystemExit(f"unknown pipeline {only_pipeline!r}; have {sorted(PIPELINES)}")
    pipelines = [only_pipeline] if only_pipeline else list(PIPELINES)
    keys = list(cfg["models"]) or list(MODEL_KEYS)
    return [cell_id(p, m) for p in pipelines for m in keys]


def is_finished(out_root: str | Path, cid: str) -> bool:
    """True only if this cell completed with every gate green.

    A scorecard that exists but failed gates is NOT finished. Skipping it would hide the one
    cell that needs attention behind a green summary line.
    """
    card = Path(out_root) / cid / "scorecard.json"
    if not card.exists():
        return False
    try:
        data = json.loads(card.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    gates = data.get("gates")
    return isinstance(gates, dict) and bool(gates) and all(gates.values())


def gate_verdict(out_root: str | Path, cid: str) -> tuple[str, int, int]:
    """(status, green, total) for a cell that has just run.

    A traceback and a failed gate BOTH exit 1, and they are not the same thing. At hour 8,
    "GATES FAILED" sends you to read gate lines that do not exist. The scorecard is the
    discriminator: run_cell writes it with a `gates` key only after the pass completed, so a
    missing scorecard means the cell died, and a red gate means it ran and the gate is the story.
    """
    card = Path(out_root) / cid / "scorecard.json"
    if not card.exists():
        return "CRASHED (no scorecard - read the traceback above)", 0, 0
    try:
        gates = json.loads(card.read_text(encoding="utf-8")).get("gates")
    except (OSError, json.JSONDecodeError) as e:
        return f"CRASHED (unreadable scorecard: {type(e).__name__})", 0, 0
    if not isinstance(gates, dict) or not gates:
        return "CRASHED (scorecard has no gates - the pass did not finish)", 0, 0
    green = sum(1 for v in gates.values() if v)
    status = "PASS" if green == len(gates) else f"GATES FAILED ({len(gates) - green} red)"
    return status, green, len(gates)


def run_one(cid: str, n: int | None, limit_corpus: int | None,
            config: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "src.runner.run_cell", "--only", cid, "--config", config]
    if n:
        cmd += ["--n", str(n)]
    if limit_corpus:
        cmd += ["--limit-corpus", str(limit_corpus)]
    # Do NOT capture output: run_cell's per-query progress is the only way to see a cell is
    # alive, and this log is what you read at hour 8. Inherit the parent's streams so nohup
    # and a live terminal both work.
    rc = subprocess.run(cmd).returncode
    return cid, rc, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every cell of the grid, one at a time.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--only", help="restrict to one pipeline (e.g. crag); default: all")
    ap.add_argument("--n", type=int, help="override eval size (pilot runs)")
    ap.add_argument("--limit-corpus", type=int, help="cap corpus paragraphs (local iteration "
                                                     "ONLY — inflates every retrieval metric)")
    ap.add_argument("--skip-finished", action="store_true",
                    help="skip cells whose scorecard exists with all gates green (resume)")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="abort the grid on the first failing cell (default: continue)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = cfg["output_root"]
    cells = all_cells(cfg, args.only)
    if args.skip_finished:
        before = len(cells)
        cells = [c for c in cells if not is_finished(out_root, c)]
        print(f"[matrix] {before - len(cells)} already finished, {len(cells)} to run")
    if not cells:
        print("[matrix] nothing to do")
        return 2

    print(f"[matrix] {len(cells)} cell(s): {' '.join(cells)}")
    print(f"[matrix] n={args.n or cfg['n']}  output={out_root}/  gpu={torch_name()}")
    if not args.n and not args.limit_corpus:
        print("[matrix] full publishable cells — no --n, no --limit-corpus")

    results: list[tuple[str, int, float, str]] = []
    for i, cid in enumerate(cells, 1):
        print(f"\n{'=' * 78}\n[matrix] {i}/{len(cells)}  {cid}\n{'=' * 78}", flush=True)
        _, rc, dt = run_one(cid, args.n, args.limit_corpus, args.config)
        status, _, _ = gate_verdict(out_root, cid)
        if rc == 0 and status != "PASS":
            status = f"rc=0 but {status} (run_cell and its scorecard disagree)"
        results.append((cid, rc, dt, status))
        print(f"[matrix] {cid}  {status}  in {dt/60:.1f} min", flush=True)
        if rc and args.stop_on_fail:
            print("[matrix] --stop-on-fail: stopping here")
            break

    print(f"\n{'=' * 78}\n[matrix] SUMMARY  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{'=' * 78}")
    for cid, _, dt, status in results:
        print(f"  {status:<12} {cid:<28} {dt/60:6.1f} min")
    passed = sum(1 for _, _, _, s in results if s == "PASS")
    crashed = sum(1 for _, _, _, s in results if s.startswith("CRASHED"))
    total_min = sum(dt for _, _, dt, _ in results) / 60
    print(f"\n{passed}/{len(results)} cells green, {total_min:.0f} GPU-min total")
    if passed and passed == len(results) and cfg["n"] and args.n is None:
        per = total_min / len(results)
        print(f"~{per:.0f} min/cell at n={cfg['n']} => {len(cells)} cells "
              f"~= {per * len(cells) / 60:.1f} GPU-h. Price the rental off that.")
    elif crashed:
        print(f"\n{crashed} cell(s) CRASHED rather than failed a gate — read the tracebacks above. "
              "A crash is an environment fault and the numbers in those cells do not exist.")
    if passed != len(results):
        print("\nCommit results/ anyway: run_cell writes manifest.json before loading any model, "
              "so a crashed cell is still attributable.")
    print("\nCommit results/ now. A reclaimed box takes the filesystem with it.")
    return 0 if passed == len(results) else 1


def torch_name() -> str:
    """Best-effort card name for the log header. Never raises: a missing torch must not stop
    the grid from launching, and run_cell's manifest records the real one anyway."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu (no GPU!)"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # No GPU, no subprocess. The two things that silently ruin an unattended grid: running
        # the cells in the wrong order (so an environment fault is discovered on the 8B cell),
        # and --skip-finished that cannot tell a finished cell from a broken one.
        import shutil

        from src.runner.run_cell import PIPELINES as P

        cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
        cells = all_cells(cfg, None)
        assert len(cells) == len(P) * len(cfg["models"]) == 6, cells
        assert len(set(cells)) == len(cells), f"duplicate cell ids: {cells}"
        assert cells == all_cells(cfg, list(P)[0]) + all_cells(cfg, list(P)[1]), \
            "--only must partition the grid, not reorder it"
        assert all_cells(cfg, "crag") == [cell_id("crag", m) for m in cfg["models"]]
        # Cheapest model first, baseline before crag: a fault should cost minutes, not hours.
        assert cells[0] == cell_id(list(P)[0], list(cfg["models"])[0]), cells
        try:
            all_cells(cfg, "nope")
            raise AssertionError("unknown pipeline must raise, not silently return the full grid")
        except SystemExit:
            pass

        tmp = Path("scratch/_matrix_selfcheck")
        shutil.rmtree(tmp, ignore_errors=True)
        cid = cells[0]
        assert not is_finished(tmp, cid), "no scorecard must not count as finished"

        def write(gates):
            d = tmp / cid
            d.mkdir(parents=True, exist_ok=True)
            (d / "scorecard.json").write_text(
                json.dumps({"cell": cid, "gates": gates}), encoding="utf-8")

        write({})
        assert not is_finished(tmp, cid), "an empty gate dict is not a passing cell"
        write({"a": True, "b": False})
        assert not is_finished(tmp, cid), "a FAILED cell must not be skipped - that hides it"
        write({"a": True, "b": True})
        assert is_finished(tmp, cid), "all-green gates must be skippable"

        # A traceback and a failed gate both exit 1. They must not be reported the same way,
        # or an unattended run tells you to read gate lines that were never written.
        assert gate_verdict(tmp, cid) == ("PASS", 2, 2), gate_verdict(tmp, cid)
        write({"a": True, "b": False, "c": False})
        st, g, t = gate_verdict(tmp, cid)
        assert st == "GATES FAILED (2 red)" and (g, t) == (1, 3), (st, g, t)
        (tmp / cid / "scorecard.json").unlink()
        assert gate_verdict(tmp, cid)[0].startswith("CRASHED"), "no scorecard == crashed"
        (tmp / cid / "scorecard.json").write_text("{ truncated", encoding="utf-8")
        assert gate_verdict(tmp, cid)[0].startswith("CRASHED"), "corrupt scorecard == crashed"
        (tmp / cid / "scorecard.json").write_text('{"cell":"x"}', encoding="utf-8")
        assert gate_verdict(tmp, cid)[0].startswith("CRASHED"), "no gates key == unfinished pass"
        shutil.rmtree(tmp, ignore_errors=True)

        print(f"matrix OK: {len(cells)} cells in cost order, "
              f"{[c.split('-')[0] for c in cells]}, skip-finished discriminates pass from fail")
        raise SystemExit(0)
    raise SystemExit(main())
