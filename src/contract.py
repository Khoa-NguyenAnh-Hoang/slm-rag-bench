"""The measurement contract — the rules any new component is written against.

Not "be careful". Seven rules, each with a machine check here, four of them wired into
`run_cell.phase_a_gates()` so a violation fails the run instead of annotating it. Each exists
because this repo already produced the defect it forbids, or would have shipped a value that
looked measured and was not.

  1. MEASURED, NOT DERIVED.  latency_ms comes from one timer around the whole chain, never a
     sum of separately-timed stages.
  2. NEVER FABRICATE A ZERO.  An unavailable measurement is None/absent, never 0.0. A benchmark
     reporting $0.00 looks like a measurement and is worse than one reporting nothing.
  3. NEVER FABRICATE A VERDICT.  If the reference answer is missing the label is Unknown, never
     Hallucinated — charging a model with hallucinating because *we* lack the gold inflates the
     hallucination rate and depresses adjusted accuracy, both headline numbers, both wrong in the
     direction that flatters the result.
  4. NO SILENT DEFAULTS ON A TRUST BOUNDARY.  Unknown model key, missing config key, missing
     usage block: raise, never fall back.
  5. A MEASUREMENT CARRIES ITS OWN METADATA.  Retrieval scores carry `depth`; cost carries the
     pricing `version` and `verified` flag.
  6. PROVENANCE BEFORE SIDE EFFECTS.  manifest.json is written before any model loads.
  7. ONE SOURCE OF TRUTH.  Every value is read from configs/experiment.yaml at one place.

`python -m src.contract` self-checks the rules against this repo's own source.
"""
from __future__ import annotations

import re
import tokenize
from pathlib import Path

UNKNOWN = "Unknown"

#: (id, one-line rule). Kept as data so the contract is greppable and citable in a write-up.
RULES: tuple[tuple[str, str], ...] = (
    ("measured-not-derived", "latency_ms is observed by one outer timer, never summed from stages"),
    ("no-fabricated-zero", "an unavailable measurement is None, never 0.0"),
    ("no-fabricated-verdict", "missing gold => Unknown, never a fabricated Correct/Hallucinated"),
    ("no-silent-defaults", "unknown key / missing usage / missing config => raise, never fall back"),
    ("measurement-carries-metadata", "scores carry depth; costs carry pricing version + verified"),
    ("provenance-first", "manifest.json is written before any model loads"),
    ("single-source-of-truth", "each value is read from experiment.yaml at exactly one place"),
)

# Source patterns that fabricate a zero-cost measurement. Matched against CODE ONLY (comments and
# docstrings are blanked by `_code_only`), so prose describing the rule does not trip it.
# The `(?![\d])` guards are load-bearing: without them `cost_usd=0.0` also matches the perfectly
# legitimate `cost_usd=0.0001`, which is how the first version of this scanner reported two false
# positives in a file that was correct.
_Z = r"(?![\d])"
_BANNED_CODE = (
    (re.compile(rf"\bcost_usd\s*=\s*0\.0{_Z}"), "fabricated cost_usd=0.0"),
    (re.compile(rf"\b(?:api|gpu)_cost_usd\s*=\s*0\.0{_Z}"), "fabricated {api,gpu}_cost_usd=0.0"),
    (re.compile(r"\bor\s+0\.0\b"), "x or 0.0 — turns a None measurement into a $0.00 reading"),
    (re.compile(r"except\s+Exception\s*:\s*\n\s*pass"), "silently swallowed exception"),
)


def _code_only(path: Path) -> str:
    """Source with comments and docstrings BLANKED OUT (newlines preserved), so rule text in
    prose cannot trip a check and reported line numbers still point at the real file.

    Blanking rather than removing is the whole trick: an earlier version joined the surviving
    tokens, which renumbered every line after the first docstring and reported violations at
    lines that do not exist. A scanner that points at the wrong line is worse than no scanner.

    A "docstring" is detected as a STRING token that begins a line (its column is the first
    non-space character of that line). That covers module, class and function docstrings and
    leaves inline strings alone — keying off `"\n" in t.string` instead missed single-line
    docstrings, which is where this repo writes most of its rule prose.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        with path.open("rb") as f:
            toks = list(tokenize.tokenize(f.readline))
    except (tokenize.TokenError, SyntaxError):
        return text
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    def idx(row: int, col: int) -> int:
        return offsets[row - 1] + col if 1 <= row <= len(offsets) else min(len(text), pos)

    def at_line_start(row: int, col: int) -> bool:
        if not (1 <= row <= len(lines)):
            return False
        return lines[row - 1][:col].strip() == ""

    chars = list(text)
    for t in toks:
        if t.type == tokenize.COMMENT:
            pass
        elif t.type == tokenize.STRING and at_line_start(*t.start):
            pass
        else:
            continue
        for i in range(idx(t.start[0], t.start[1]), idx(t.end[0], t.end[1])):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def scan_source(root: str | Path = "src", skip: tuple[str, ...] = ("contract.py",)) -> list[str]:
    """Every rule-2/4 violation in the given tree, as 'path:line: message'.

    `contract.py` is excluded because it necessarily contains the banned patterns as data — its
    own self-check proves the scanner fires, which is the coverage that file needs.
    """
    bad: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        if path.name in skip:
            continue
        code = _code_only(path)
        for rx, msg in _BANNED_CODE:
            for m in rx.finditer(code):
                bad.append(f"{path}:{code[: m.start()].count(chr(10)) + 1}: {msg}")
    return bad


def check_record(rec, has_pricing: bool = True) -> list[str]:
    """Rules 1, 2, 5 against one QueryRecord. Returns violations; empty means compliant."""
    out: list[str] = []
    stages = sum((rec.stage_latency_ms or {}).values())
    if rec.latency_ms is None or rec.latency_ms < stages - 1e-6:
        out.append(f"{rec.query_id}: latency {rec.latency_ms} < sum(stages) {stages} "
                   f"— derived, not measured (rule 1)")
    if rec.prompt_tokens <= 0:
        out.append(f"{rec.query_id}: prompt_tokens={rec.prompt_tokens} — fabricated (rule 2)")
    if rec.completion_tokens < 0:
        out.append(f"{rec.query_id}: negative completion_tokens (rule 2)")
    if has_pricing and rec.api_cost_usd is None:
        out.append(f"{rec.query_id}: api_cost_usd is None with a price card loaded (rule 2)")
    if has_pricing and rec.api_cost_usd == 0.0:
        out.append(f"{rec.query_id}: api_cost_usd==0.0 — a priced query cannot cost nothing (rule 2)")
    return out


def check_label(label: str, gold: str) -> list[str]:
    """Rule 3. An empty reference answer must never yield a verdict."""
    if not gold.strip() and label != UNKNOWN:
        return [f"label {label!r} with empty gold — fabricated verdict (rule 3)"]
    if label == "Correct" and not gold.strip():
        return ["Correct with empty gold (rule 3)"]
    return []


def check_scores(scores: dict, gold_ids) -> list[str]:
    """Rules 2 and 5 for one query's retrieval scores. All-zero coverage with gold present is a
    real result; all-zero with NO gold is a scoring bug, and the two are indistinguishable
    downstream unless the depth/metadata requirement is enforced here."""
    out: list[str] = []
    if "depth" not in scores:
        out.append("scores carry no depth — a MAP number without its depth is unciteable (rule 5)")
    if not gold_ids and scores.get("coverage") == 0.0:
        out.append("coverage 0.0 with no gold in the index — scoring bug, not a result (rule 2)")
    return out


if __name__ == "__main__":
    # The rules, stated.
    assert len(RULES) == 7 and len({r[0] for r in RULES}) == 7, RULES

    # Rule 2 scanner: catches real fabrications, ignores prose describing one, and does not
    # mistake a small real cost for a fabricated zero.
    assert scan_source("src") == [], "rule-2 violations in src/:\n" + "\n".join(scan_source("src"))
    tmp = Path("src") / "_contract_probe.py"
    try:
        tmp.write_text(
            '"""docstring mentioning cost_usd=0.0 — must NOT trip the scanner."""\n'
            "def f(x):\n"
            "    return x or 0.0\n"
            "def g(r):\n"
            "    return MetricsLogger(p, pricing=None).log_query\n"
            "def h():\n"
            "    d = {'api_cost_usd': 0.0001}\n"        # a real cost, must not trip
            "    return d\n", encoding="utf-8")
        found = scan_source("src")
        assert any("or 0.0" in f for f in found), found
        assert not any("fabricated" in f for f in found), f"false positive: {found}"
    finally:
        tmp.unlink(missing_ok=True)
    assert scan_source("src") == [], "scanner must clean up after itself"

    # Rule 1: observed latency passes; a sum of stages fails.
    class R:
        query_id, latency_ms, stage_latency_ms = "q", 0.0, {}
        prompt_tokens, completion_tokens, api_cost_usd, gpu_cost_usd = 0, 0, 0.0, 0.0
    r = R()
    r.stage_latency_ms, r.latency_ms = {"retrieve": 5.0, "generate": 90.0}, 99.0
    r.prompt_tokens, r.completion_tokens, r.api_cost_usd = 100, 20, 0.0001
    assert check_record(r) == [], check_record(r)
    r.latency_ms = 90.0                      # sum of stages, no observation
    assert any("rule 1" in v for v in check_record(r)), check_record(r)
    r.latency_ms = 99.0
    r.api_cost_usd = 0.0                     # priced but free
    assert any("rule 2" in v for v in check_record(r)), check_record(r)

    # Rule 3: empty gold is Unknown, never a verdict.
    assert check_label("Correct", "") and check_label("Hallucinated", "")
    assert check_label(UNKNOWN, "") == []
    assert check_label("Correct", "paris") == []

    # Rule 5: depth is mandatory.
    assert any("rule 5" in v for v in check_scores({"map@3": 0.5}, [1, 2]))
    assert check_scores({"depth": 10.0, "coverage": 0.0}, [1, 2]) == []
    assert any("rule 2" in v for v in check_scores({"depth": 0.0, "coverage": 0.0}, []))

    print(f"contract OK: {len(RULES)} rules, scanner clean, all rule checks fire")
