"""Per-query cost, latency and token logger.

Records mean/P95 latency, per-stage latency, token counts, and TWO labelled cost numbers:
  api_cost_usd — token counts × a hosted-API rate card (configs/pricing.yaml). A proxy that
                 makes cost comparable to commercially served models.
  gpu_cost_usd — measured wall-clock × a stated $/GPU-hour. What the run actually cost.

Neither defaults to 0.0: a benchmark reporting zero cost looks like a measurement, and is worse
than one reporting nothing. Both are None when no price card is supplied, and summary() reports
how many queries carry each.

Per-stage latency exists because the reranker's own cost is only visible when rerank is timed as
its own stage — fused into retrieval it disappears entirely.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class QueryRecord:
    query_id: str
    model_name: str
    latency_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    ttft_ms: float | None = None
    tok_per_s: float | None = None
    api_cost_usd: float | None = None
    gpu_cost_usd: float | None = None
    extra: dict = field(default_factory=dict)


class MetricsLogger:
    """Accumulates per-query records in memory and appends each as one JSONL line.

    `pricing` is the resolved per-model entry from configs/pricing.yaml:
        {api_usd_per_1m: {prompt: float, completion: float}, gpu_usd_per_hour: float}
    """

    def __init__(self, out_jsonl: str | Path, pricing: dict | None = None) -> None:
        self.out_path = Path(out_jsonl)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[QueryRecord] = []
        self.pricing = pricing

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> tuple[float | None, float | None]:
        if not self.pricing:
            return None, None
        rate = self.pricing.get("api_usd_per_1m") or {}
        api = ((prompt_tokens * (rate.get("prompt", 0.0))
                + completion_tokens * (rate.get("completion", 0.0))) / 1e6)
        return api, None       # gpu cost is filled by the caller, which owns the wall clock

    def log_query(
        self,
        query_id: str,
        latency_ms: float | None,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
        stage_latency_ms: dict[str, float] | None = None,
        ttft_ms: float | None = None,
        gpu_seconds: float | None = None,
        **extra,
    ) -> QueryRecord:
        api_usd, gpu_usd = self._cost(prompt_tokens, completion_tokens)
        if gpu_usd is None and gpu_seconds is not None and self.pricing:
            gpu_usd = gpu_seconds * (self.pricing.get("gpu_usd_per_hour", 0.0) / 3600.0)
        tok_per_s = (completion_tokens / (latency_ms / 1000.0)
                     if latency_ms and latency_ms > 0 and completion_tokens else None)
        rec = QueryRecord(
            query_id=query_id, model_name=model_name, latency_ms=latency_ms,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            stage_latency_ms=stage_latency_ms or {}, ttft_ms=ttft_ms, tok_per_s=tok_per_s,
            api_cost_usd=api_usd, gpu_cost_usd=gpu_usd, extra=extra,
        )
        self.records.append(rec)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec

    @staticmethod
    def _stats(values: Iterable[float]) -> dict[str, float]:
        v = np.asarray([x for x in values if x is not None], dtype=float)
        if v.size == 0:
            return {"mean": None, "p95": None, "n": 0}
        return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "n": int(v.size)}

    @staticmethod
    def _sum(values: Iterable[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return float(sum(vals)) if vals else None

    def summary(self) -> dict:
        stages = {s: self._stats(r.stage_latency_ms[s] for r in self.records
                                 if s in r.stage_latency_ms)
                  for s in sorted({k for r in self.records for k in r.stage_latency_ms})}
        return {
            "n_queries": len(self.records),
            "latency_ms": self._stats(r.latency_ms for r in self.records),   # mean + P95
            "stage_latency_ms": stages,                                      # broken down per stage
            "ttft_ms": self._stats(r.ttft_ms for r in self.records),
            "tok_per_s": self._stats(r.tok_per_s for r in self.records),
            "prompt_tokens": int(sum(r.prompt_tokens for r in self.records)),
            "completion_tokens": int(sum(r.completion_tokens for r in self.records)),
            "total_tokens": int(sum(r.prompt_tokens + r.completion_tokens for r in self.records)),
            "api_cost_usd": self._sum(r.api_cost_usd for r in self.records),
            "gpu_cost_usd": self._sum(r.gpu_cost_usd for r in self.records),
            "n_with_api_cost": sum(r.api_cost_usd is not None for r in self.records),
            "n_with_gpu_cost": sum(r.gpu_cost_usd is not None for r in self.records),
        }


class stage_timer:
    """with-block millisecond timer: `with stage_timer(d, "rerank"):` -> d['rerank'] = ms."""

    def __init__(self, sink: dict[str, float], key: str) -> None:
        self.sink, self.key = sink, key

    def __enter__(self) -> "stage_timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.sink[self.key] = (time.perf_counter() - self._t0) * 1000.0


def load_pricing(model_key: str, path: str | Path = "configs/pricing.yaml") -> dict:
    """Resolve the price card for one model key (configs/pricing.yaml).

    Raises on an unknown key rather than defaulting to 0.0: a mis-typed model name must never
    quietly zero the thesis's cost axis. The `verified` flag and pricing `version` ride along
    into the scorecard, so an unverified rate card is visible in the published artifact.
    """
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        p = yaml.safe_load(f)
    providers = p.get("providers", {})
    if model_key not in providers:
        raise KeyError(f"no price card for {model_key!r}; have {sorted(providers)} — "
                       "refusing to report $0.00")
    return {
        "api_usd_per_1m": providers[model_key]["api_usd_per_1m"],
        "gpu_usd_per_hour": p.get("gpu_usd_per_hour", 0.0),
        "version": p.get("version"),
        "verified": providers[model_key].get("verified", False),
    }


if __name__ == "__main__":
    import tempfile

    lg = MetricsLogger(Path(tempfile.gettempdir()) / "selfcheck.jsonl")
    for i in range(20):
        lg.log_query(f"q{i}", latency_ms=100 + i * 10, prompt_tokens=200,
                     completion_tokens=50, model_name="selfcheck",
                     stage_latency_ms={"retrieve": 5.0}, ttft_ms=40.0)
    s = lg.summary()
    # Known-answer check: 20 values 100..290 -> mean 195; numpy p95 = 280.5
    assert s["n_queries"] == 20 and abs(s["latency_ms"]["mean"] - 195.0) < 1e-6, s
    assert abs(s["latency_ms"]["p95"] - 280.5) < 1e-6, s["latency_ms"]
    assert s["prompt_tokens"] == 4000
    # No pricing card => cost is None, never a fabricated 0.0 (contract rule 2).
    assert s["api_cost_usd"] is None and s["n_with_api_cost"] == 0, s

    priced = MetricsLogger(Path(tempfile.gettempdir()) / "priced.jsonl",
                           pricing={"api_usd_per_1m": {"prompt": 0.15, "completion": 0.60},
                                    "gpu_usd_per_hour": 0.90})
    priced.log_query("q0", latency_ms=1000, prompt_tokens=1_000_000, completion_tokens=0,
                     model_name="m", gpu_seconds=3600)
    p = priced.summary()
    assert abs(p["api_cost_usd"] - 0.15) < 1e-9 and abs(p["gpu_cost_usd"] - 0.90) < 1e-9, p
    assert p["n_with_api_cost"] == 1 and p["n_with_gpu_cost"] == 1

    # Price-card resolution must fail loudly on a bad key, never silently price at zero.
    # Asserted against the SHIPPED card, and only on properties that must hold whatever the
    # current rates are: non-zero rates, a version string, a real bool. An earlier version of this
    # line asserted `verified is False` — pinning a state of the world, so verifying the rates
    # turned a green check red. The flag is verified by reading configs/pricing.yaml, not here.
    card = load_pricing("llama3.1-8b")
    assert card["api_usd_per_1m"]["prompt"] > 0, card
    assert card["api_usd_per_1m"]["completion"] > 0, card
    assert isinstance(card["verified"], bool) and card["version"], card
    assert card["gpu_usd_per_hour"] > 0, "contract rule 2: a rental is never free"
    try:
        load_pricing("not-a-model")
        raise SystemExit("FAIL: unknown model key must raise, not return $0.00")
    except KeyError:
        pass
    print("metrics_logger OK:", s["latency_ms"], "| priced:", p["api_cost_usd"], p["gpu_cost_usd"],
          "| card:", card["version"], "verified=" + str(card["verified"]))
