"""Judge probe: 5 canned samples through the FULL serve + ragas 0.4.3 path.

Run BEFORE any cell pass:
    uv run python -m src.runner.judge_probe --config configs/experiment.yaml --model-key qwen3-4b

Two gates, in this order, because they fail for different reasons and cost different amounts:

  1. STREAM/USAGE — one streaming completion must return a usage block and a positive TTFT.
     The token counts feed the ENTIRE cost axis, so a missing usage block is not cosmetic: the
     probe raises rather than letting zeros flow into pricing.
  2. FAITHFULNESS — >= 4/5 non-NaN through the ragas collections API. One NaN is normal (the
     abstention branch has no statements to support).

Exit 0 on pass, 1 on fail, so a Colab shell cell can gate on it: `!python -m src.runner.judge_probe ...`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROBE_SAMPLES = [
    {"question": "Who painted the Mona Lisa?",
     "contexts": ["The Mona Lisa was painted by Leonardo da Vinci in the early 1500s."],
     "answer": "Leonardo da Vinci painted the Mona Lisa."},
    {"question": "Where is the Eiffel Tower?",
     "contexts": ["The Eiffel Tower stands on the Champ de Mars in Paris, France."],
     "answer": "The Eiffel Tower is in Paris."},
    {"question": "What year was the Starry Night painted?",
     "contexts": ["Vincent van Gogh painted The Starry Night in June 1889."],
     "answer": "The Starry Night was painted in 1889."},
    {"question": "What is the capital of Japan?",
     "contexts": ["Tokyo is the capital and largest city of Japan."],
     "answer": "Tokyo is the capital of Japan."},
    {"question": "Who discovered penicillin?",
     "contexts": [""],                       # abstention branch: empty context, expect NaN/0
     "answer": "The context does not provide sufficient information to answer the question."},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model-key", default="qwen3-4b")
    ap.add_argument("--min-pass", type=int, default=None,
                    help="default: judge.probe_min_pass in the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model_key not in cfg["models"]:
        raise SystemExit(f"unknown --model-key {args.model_key!r}; have {sorted(cfg['models'])}")
    cfg["_model_key"] = args.model_key
    min_pass = args.min_pass if args.min_pass is not None \
        else int(cfg["judge"].get("probe_min_pass", 4))

    from src.runner.run_cell import ragas_scores
    from src.runner.serve import (build_embeddings, build_judge, serve_client, serve_generate,
                                  server_from_config, serving)

    with serving(server_from_config(cfg, "generator")) as server:
        client = serve_client(server)

        # Gate 1: streaming + usage block + TTFT. Cheap (one call) and it must come first.
        text, p_tok, c_tok, ttft = serve_generate(
            client, server.model, "Say 'ok'.", max_new_tokens=8)
        if not (p_tok > 0 and c_tok > 0 and ttft > 0):
            print(f"PROBE FAIL: streaming/usage — prompt={p_tok} completion={c_tok} ttft={ttft}")
            print("  the cost axis depends on this; fix the client/stream_options before running")
            return 1
        print(f"stream OK: prompt={p_tok} completion={c_tok} ttft={ttft:.0f}ms {text!r}")

        judge = build_judge(server, cfg["judge"]["mode"])
        out = ragas_scores(PROBE_SAMPLES, judge, build_embeddings())

    print(json.dumps(out, indent=2))
    scored = out.get("faithfulness_n", 0)
    ok = scored >= min_pass
    print(f"PROBE {'PASS' if ok else 'FAIL'}: faithfulness_n={scored} (need >={min_pass})")
    if not ok and cfg["judge"].get("mode") == "JSON":
        print("  first thing to try: judge.mode JSON -> MD_JSON in the config")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
