"""Command-line surface (spec §13): ``python -m groundcheck.cli check ...``.

```
check  --example FILE | (--source-file F --answer-file F)  [-n N] [--json]
```

The headline score is formatted with **integer truncation** (``int(score*100)``),
not ``round`` — ``0.625`` prints ``62%``. This is deliberate and load-bearing: it
keeps the CLI on the same rule as the web frontend (``Math.floor``), where
``Math.round(62.5)`` would wrongly read ``63`` (see Split 09). Expected conditions
(missing key, bad input file) print a clean one-line message and a non-zero exit —
never a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import DEFAULT_N_RUNS, N_RUNS_MAX, N_RUNS_MIN, VERDICT_WORDS
from .models import FaithfulnessReport
from .pipeline import check

# Exit codes (0 = success). Distinct codes make scripting / tests precise.
_EXIT_OK = 0
_EXIT_USAGE = 2  # bad / missing input
_EXIT_NO_KEY = 3  # a real provider was selected but no key is configured


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    parser.print_help()
    return _EXIT_USAGE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundcheck",
        description="Verify an answer's claims against a source document.",
    )
    sub = parser.add_subparsers(dest="command")

    check_p = sub.add_parser("check", help="check one answer against one source")
    check_p.add_argument(
        "--example",
        metavar="FILE",
        help='JSON file with {"source": ..., "answer": ...}',
    )
    check_p.add_argument("--source-file", metavar="F", help="read the SOURCE from this file")
    check_p.add_argument("--answer-file", metavar="F", help="read the ANSWER from this file")
    check_p.add_argument(
        "-n",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"grounding runs per claim (majority vote); default {DEFAULT_N_RUNS}",
    )
    check_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the full FaithfulnessReport as JSON",
    )
    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        source, answer = _load_inputs(args)
    except _InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    n = _clamp_n(args.n)

    try:
        report = check(source, answer, n=n)
    except RuntimeError as exc:
        # Expected: a real provider was selected but no key is configured. The
        # provider's message already names the fix (set a key or GROUNDCHECK_LLM=mock).
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Set an API key, or run in demo mode with GROUNDCHECK_LLM=mock.",
            file=sys.stderr,
        )
        return _EXIT_NO_KEY

    if args.as_json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)
    return _EXIT_OK


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #


class _InputError(Exception):
    """A bad/missing CLI input, surfaced as a clean message (no traceback)."""


def _load_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if args.example:
        path = Path(args.example)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise _InputError(f"example file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise _InputError(f"{path} is not valid JSON: {exc}") from None
        if not isinstance(data, dict) or "source" not in data or "answer" not in data:
            raise _InputError(f'{path} must be a JSON object with "source" and "answer"')
        return str(data["source"]), str(data["answer"])

    if args.source_file and args.answer_file:
        try:
            source = Path(args.source_file).read_text(encoding="utf-8")
            answer = Path(args.answer_file).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise _InputError(str(exc)) from None
        return source, answer

    raise _InputError("provide --example FILE, or both --source-file and --answer-file")


def _clamp_n(n: int) -> int:
    """Keep N within the supported stepper range (UI spec §B), warning if clamped."""
    clamped = max(N_RUNS_MIN, min(N_RUNS_MAX, n))
    if clamped != n:
        print(
            f"note: N={n} clamped to {clamped} (supported range {N_RUNS_MIN}–{N_RUNS_MAX})",
            file=sys.stderr,
        )
    return clamped


# --------------------------------------------------------------------------- #
# Human-readable rendering
# --------------------------------------------------------------------------- #


def _print_report(report: FaithfulnessReport) -> None:
    print(f"Faithfulness: {_score_text(report)}  ({report.n_supported}/{report.n_claims} grounded)")
    if report.n_low_confidence:
        print(f"  {report.n_low_confidence} to review (confidence < 1.0)")
    if report.n_refused:
        print(f"  {report.n_refused} refused by the judge -> counted as NOT_ENOUGH_INFO")
    print()

    # Output is intentionally ASCII-only so it never trips a strict Windows console
    # encoding (the spec's middot is a UI nicety, not a CLI contract).
    for i, claim in enumerate(report.claims, start=1):
        word = VERDICT_WORDS[claim.label]
        review = " (review)" if claim.confidence < 1.0 else ""
        print(
            f"[{i}] {word}  |  votes {_format_votes(claim.votes)}  |  "
            f"confidence {claim.confidence:.2f}{review}"
        )
        print(f"    claim: {claim.claim}")
        if claim.supporting_span:
            print(f"    span:  {claim.supporting_span}")
        print(f"    why:   {claim.rationale}")
    if report.unlocated_sentences:
        print()
        print(
            f"note: {len(report.unlocated_sentences)} sentence(s) could not be located "
            "in the answer (shown above, not highlighted)"
        )
    print()
    print(
        f"cost ${report.cost_usd:.4f} | {report.latency_s:.1f}s | "
        f"N={report.n_runs} | prompt {report.prompt_version}"
    )


def _score_text(report: FaithfulnessReport) -> str:
    """``62%`` for 0.625 (integer truncation); ``N/A`` when there are no claims."""
    score = report.faithfulness_score
    return "N/A" if score is None else f"{int(score * 100)}%"


def _format_votes(votes: dict[str, int]) -> str:
    """Compact vote spread, most-voted label first (e.g. ``SUPPORTED 2, NOT_ENOUGH_INFO 1``)."""
    if not votes:
        return "(none)"
    ordered = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{label} {count}" for label, count in ordered)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
