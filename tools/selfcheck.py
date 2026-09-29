"""Run every CPU self-check in one command. No GPU, no network (after the first HotpotQA pull).

    uv run python -m tools.selfcheck

Two Phase 1 self-checks turned out never to have executed — `crag_module`'s smoke test raised
TypeError on its first honest run, and `retrieval_metrics` did not exist. So this exists to make
"the checks pass" a single fact rather than a claim: one command, one exit code, every module
that carries an `if __name__ == "__main__"` assert. Non-zero exit if any module fails.
"""
from __future__ import annotations

import subprocess
import sys
import time

# Order is cheap-before-expensive: pure-stdlib modules first so a syntax error surfaces in
# seconds instead of after a 400 MB model download.
CHECKS = [
    "src.monitor.retrieval_metrics",
    "src.monitor.metrics_logger",
    "src.pipeline",
    "src.runner.run_cell",           # needs --self-check; it is the only CLI with a flag
    "src.data.hotpot_loader",        # downloads HotpotQA on first run
    "src.retrieval.hybrid_reranker",  # downloads MPNet + MiniLM on first run
    "src.evaluator.crag_module",      # downloads flan-t5-base on first run
]
FLAGS = {"src.runner.run_cell": ["--self-check"]}


def main() -> int:
    failed, slow = [], []
    for mod in CHECKS:
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", mod, *FLAGS.get(mod, [])],
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0
        if not ok:
            failed.append(mod)
        if dt > 60:
            slow.append((mod, dt))
        tail = next((l for l in reversed((r.stdout or "").splitlines())
                     if "OK" in l or "NOTE" in l), "")
        print(f"{'PASS' if ok else 'FAIL'}  {dt:6.1f}s  {mod:<32} {tail[:78]}")
        if not ok:
            print("\n".join((r.stderr or "").strip().splitlines()[-8:]), file=sys.stderr)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed"
          + (f" — FAILED: {failed}" if failed else ""))
    if slow:
        print("note: first run downloads models; later runs are fast:\n  "
              + "\n  ".join(f"{m} {d:.0f}s" for m, d in slow))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
