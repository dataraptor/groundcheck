"""The two-tier meta-eval harness — ``python -m eval.run`` (spec §11, §13).

This is the §13 ``eval`` surface, shipped in the **eval layer** (not as a
``groundcheck.cli`` subcommand) so the gold datasets stay out of ``core`` — see
the SPLIT-07 divergence note. It imports the orchestrator **directly** (no HTTP,
spec §4) and delegates **all** metric arithmetic to :mod:`groundcheck.metrics`
(this module contains no P/R/F1/κ math of its own).

Two tiers, each with clean ground truth (spec §11):

* **Tier 1 — grounding-judge accuracy.** Feed each fixed ``{source, claim,
  gold_label}`` triple straight into :func:`groundcheck.ground` (no decomposition,
  so the detector label aligns 1:1 with gold). Report per-class P/R/F1, macro-F1,
  accuracy and Cohen's κ via :func:`groundcheck.tier1_report`. Decomposition
  variance deliberately does **not** enter here — it belongs to Tier 2.
* **Tier 2 — end-to-end answer-level detection.** Run the full
  :func:`groundcheck.check` and apply the §11 ``predicted_unfaithful`` rule over a
  τ sweep, scored with :func:`groundcheck.binary_prf` (positive = "unfaithful").

Grounding is non-deterministic (spec §9), so every tier runs **R repeats** and is
reported as **mean ± spread**, never a single number. Each dataset is split into a
**held-in** slice and a **frozen** (~20%) slice, reported separately; the frozen
slice — too small for stable per-class F1 — is reported as **accuracy + κ only**
(spec §11 "Size honesty"). Every run is persisted to ``runs/<ts>.jsonl`` with a
header recording ``prompt_version`` + both model IDs + ``n`` + ``R`` + ``mock_mode``
so it is fully reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ValidationError

from groundcheck import binary_prf, check, ground, tier1_report
from groundcheck.config import DECOMPOSE_MODEL, DEFAULT_N_RUNS, GROUND_MODEL
from groundcheck.llm import LLMProvider, get_provider
from groundcheck.models import Label
from groundcheck.prompts import PROMPT_VERSION

logger = logging.getLogger("eval.run")

# The 3-class Tier-1 label set (order is the metrics' column/row order).
LABELS: list[str] = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]

# The Tier-2 τ sweep (operating point first). Any non-SUPPORTED claim flags the
# answer at τ=1.0; the lower thresholds show the precision/recall trade-off (§11).
TAU_SWEEP: tuple[float, ...] = (1.0, 0.9, 0.8)
DEFAULT_REPEATS = 3

# The five Tier-2 answer buckets (Appendix B.2).
TIER2_BUCKETS = (
    "fully_grounded",
    "single_fabricated_stat",
    "direct_contradiction",
    "faithful_paraphrase",
    "mixed",
)

_HERE = Path(__file__).resolve().parent
DATASETS_DIR = _HERE / "datasets"
RUNS_DIR = _HERE / "runs"


# --------------------------------------------------------------------------- #
# Dataset item contracts + loader
# --------------------------------------------------------------------------- #


class Tier1Item(BaseModel):
    """One fixed grounding triple: ``{source, claim, gold_label}`` (Appendix B.1)."""

    id: str
    topic: str
    source: str
    claim: str
    gold_label: Label  # SUPPORTED | CONTRADICTED | NOT_ENOUGH_INFO (else → skipped)
    frozen: bool = False
    provenance: str = ""


class Tier2Item(BaseModel):
    """One end-to-end answer case: ``{source, answer, gold_is_faithful}`` (B.2)."""

    id: str
    topic: str
    source: str
    answer: str
    gold_is_faithful: bool
    bucket: Literal[
        "fully_grounded",
        "single_fabricated_stat",
        "direct_contradiction",
        "faithful_paraphrase",
        "mixed",
    ]
    frozen: bool = False
    provenance: str = ""


@dataclass
class LoadedDataset:
    """A parsed, per-item-validated dataset plus any items that were skipped.

    ``items`` are validated :class:`Tier1Item` / :class:`Tier2Item` objects;
    ``skipped`` is ``[(raw_entry, reason), ...]`` for entries that failed schema
    validation (a malformed item is flagged + skipped, never silently dropped and
    never crashing the run — spec §17 / SPLIT-07 error discipline).
    """

    kind: str  # "tier1" | "tier2"
    items: list[Any]
    skipped: list[tuple[Any, str]] = field(default_factory=list)


def _detect_kind(raw: list) -> str:
    """Infer ``tier1`` vs ``tier2`` from the keys present across all entries."""
    keys: set[str] = set()
    for entry in raw:
        if isinstance(entry, dict):
            keys |= set(entry)
    if "gold_label" in keys or "claim" in keys:
        return "tier1"
    if "gold_is_faithful" in keys or "answer" in keys:
        return "tier2"
    raise ValueError(
        "cannot detect dataset kind: no 'claim'/'gold_label' (tier1) or "
        "'answer'/'gold_is_faithful' (tier2) keys found"
    )


def load_dataset(path: str | Path, kind: Optional[str] = None) -> LoadedDataset:
    """Parse + validate a YAML gold dataset (the single loader for harness + tests).

    Each entry is validated against its tier's schema (required fields, valid
    ``gold_label`` / ``bucket`` / bools). A malformed entry is **flagged** into
    ``skipped`` with a reason rather than crashing the run. Structural problems
    that cannot be per-item skipped — a non-list / empty file or a **duplicate id**
    — raise :class:`ValueError` (they would corrupt the gold↔pred alignment).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: dataset must be a non-empty YAML list of items")

    kind = kind or _detect_kind(raw)
    model = Tier1Item if kind == "tier1" else Tier2Item

    items: list[Any] = []
    skipped: list[tuple[Any, str]] = []
    seen_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            skipped.append((entry, "entry is not a mapping"))
            continue
        try:
            item = model(**entry)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            reason = f"schema invalid: {first.get('loc', ('?',))[0]}: {first.get('msg', exc)}"
            skipped.append((entry, reason))
            continue
        if item.id in seen_ids:
            raise ValueError(f"{path}: duplicate id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)

    return LoadedDataset(kind=kind, items=items, skipped=skipped)


def split_slice(items: list[Any], which: str) -> dict[str, list[Any]]:
    """Partition items into the requested slice(s) by the ``frozen`` flag.

    ``held-in`` = ``frozen: false`` items; ``frozen`` = ``frozen: true`` items;
    ``all`` returns both, reported separately (spec §11 gating). Empty slices are
    omitted so a tier with no frozen items doesn't try to score an empty set.
    """
    held_in = [it for it in items if not it.frozen]
    frozen = [it for it in items if it.frozen]
    out: dict[str, list[Any]] = {}
    if which in ("held-in", "all") and held_in:
        out["held-in"] = held_in
    if which in ("frozen", "all") and frozen:
        out["frozen"] = frozen
    return out


# --------------------------------------------------------------------------- #
# mean ± spread aggregation (distributional reporting, spec §11)
# --------------------------------------------------------------------------- #


def _mean_spread(values: list[float]) -> dict[str, float]:
    """``mean`` + spread (``min`` / ``max`` / population ``stdev``) over R repeats."""
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Tier 2 prediction rule (the two §11 corner cases live here)
# --------------------------------------------------------------------------- #


def predicted_unfaithful(report, tau: float) -> bool:
    """The §11 detector rule: is this answer flagged as unfaithful at threshold τ?

    ``predicted_unfaithful = (n_contradicted > 0) OR (score is not None AND
    score < τ)``. Two corner cases are pinned here so they are never re-derived
    mid-build:

    * **(a)** the ``n_contradicted > 0`` clause flags a contradiction **regardless
      of τ**. At τ=1.0 it is redundant (any contradiction already drops the score
      below 1.0), but it is **load-bearing at τ<1.0**, where it keeps a single hard
      contradiction from being averaged away by many supported claims.
    * **(b)** a ``score is None`` answer (0 checkable claims) is predicted
      **faithful** — both sub-clauses are false. Tier-2 gold cases therefore always
      carry ≥1 checkable claim (§17 / §11 corner case (b)), or are excluded.
    """
    return report.n_contradicted > 0 or (
        report.faithfulness_score is not None and report.faithfulness_score < tau
    )


# --------------------------------------------------------------------------- #
# Tier 1 — grounding-judge accuracy
# --------------------------------------------------------------------------- #


@dataclass
class Tier1SliceResult:
    slice_name: str
    n_items: int
    repeats: int
    per_repeat: list[dict]  # one tier1_report() per repeat
    aggregate: dict  # mean ± spread per metric
    refusal_affected: int
    item_lines: list[dict]  # per-item JSONL prediction records


def run_tier1(
    items: list[Tier1Item],
    *,
    n: int,
    repeats: int,
    provider: LLMProvider,
    slice_name: str = "held-in",
) -> Tier1SliceResult:
    """Ground each fixed claim ``n`` times over ``R`` repeats; aggregate the §11 metrics.

    No decomposition — the claim is fixed, so the detector label
    (:attr:`GroundOutcome.label`) aligns 1:1 with ``gold_label`` (spec §11 Tier 1).
    Each repeat produces a full :func:`groundcheck.tier1_report`; the slice result
    carries the per-repeat reports plus the mean ± spread across them.
    """
    golds = [it.gold_label for it in items]
    per_repeat: list[dict] = []
    item_lines: list[dict] = []
    refusal_affected = 0

    for r in range(1, repeats + 1):
        preds: list[str] = []
        for it in items:
            outcome = ground(it.source, it.claim, n=n, provider=provider)
            preds.append(outcome.label)
            if outcome.n_refused_runs > 0:
                refusal_affected += 1
            item_lines.append(
                {
                    "kind": "prediction",
                    "tier": "1",
                    "slice": slice_name,
                    "id": it.id,
                    "repeat": r,
                    "gold": it.gold_label,
                    "predicted": outcome.label,
                    "confidence": outcome.confidence,
                    "refused": outcome.refused,
                    "n_refused_runs": outcome.n_refused_runs,
                }
            )
        per_repeat.append(tier1_report(golds, preds, LABELS))

    return Tier1SliceResult(
        slice_name=slice_name,
        n_items=len(items),
        repeats=repeats,
        per_repeat=per_repeat,
        aggregate=_aggregate_tier1(per_repeat),
        refusal_affected=refusal_affected,
        item_lines=item_lines,
    )


def _aggregate_tier1(reports: list[dict]) -> dict:
    """Mean ± spread across repeats for macro-F1 / accuracy / κ and per-class P/R/F1."""
    return {
        "macro_f1": _mean_spread([rep["macro_f1"] for rep in reports]),
        "accuracy": _mean_spread([rep["accuracy"] for rep in reports]),
        "kappa": _mean_spread([rep["kappa"] for rep in reports]),
        "per_class": {
            c: {
                "precision": _mean_spread([rep["per_class"][c]["precision"] for rep in reports]),
                "recall": _mean_spread([rep["per_class"][c]["recall"] for rep in reports]),
                "f1": _mean_spread([rep["per_class"][c]["f1"] for rep in reports]),
                "support": reports[0]["per_class"][c]["support"],
            }
            for c in LABELS
        },
        "n": reports[0]["n"],
    }


def _tier1_slice_summary(result: Tier1SliceResult, *, frozen: bool) -> dict:
    """Trim a Tier-1 slice to its reportable headline (spec §11 "Size honesty").

    The **held-in** slice headlines the full set (macro-F1 + accuracy + κ +
    per-class). The **frozen** slice (~9 items) headlines **accuracy + κ only** —
    per-class / macro-F1 are too noisy at that size to report — with an explicit
    ``n≈9, wide interval`` caption.
    """
    if frozen:
        return {
            "slice": result.slice_name,
            "n": result.n_items,
            "repeats": result.repeats,
            "accuracy": result.aggregate["accuracy"],
            "kappa": result.aggregate["kappa"],
            "caption": (
                f"frozen slice n={result.n_items} (~9, wide interval) -- accuracy + kappa "
                "only; per-class / macro-F1 too noisy to headline at this size (spec 11)"
            ),
        }
    return {
        "slice": result.slice_name,
        "n": result.n_items,
        "repeats": result.repeats,
        "macro_f1": result.aggregate["macro_f1"],
        "accuracy": result.aggregate["accuracy"],
        "kappa": result.aggregate["kappa"],
        "per_class": result.aggregate["per_class"],
    }


# --------------------------------------------------------------------------- #
# Tier 2 — end-to-end answer-level detection
# --------------------------------------------------------------------------- #


@dataclass
class Tier2SliceResult:
    slice_name: str
    n_items: int
    repeats: int
    aggregate: dict  # {str(tau): {metric: mean_spread}}
    refusal_affected: int
    item_lines: list[dict]


def run_tier2(
    items: list[Tier2Item],
    *,
    n: int,
    repeats: int,
    provider: LLMProvider,
    taus: tuple[float, ...] = TAU_SWEEP,
    slice_name: str = "held-in",
) -> Tier2SliceResult:
    """Run full :func:`check` per answer over ``R`` repeats; binary P/R per τ.

    The positive class is **unfaithful**, so the gold vector is
    ``not gold_is_faithful``. For each τ the per-item prediction comes from
    :func:`predicted_unfaithful`; :func:`groundcheck.binary_prf` turns the gold/pred
    vectors into precision/recall/F1/accuracy, aggregated mean ± spread over repeats.
    """
    gold_unfaithful = [not it.gold_is_faithful for it in items]
    per_repeat: list[dict[float, dict]] = []
    item_lines: list[dict] = []
    refusal_affected = 0

    for r in range(1, repeats + 1):
        reports = []
        for it in items:
            report = check(it.source, it.answer, n=n, provider=provider)
            reports.append(report)
            if report.n_refused > 0:
                refusal_affected += 1
            item_lines.append(
                {
                    "kind": "prediction",
                    "tier": "2",
                    "slice": slice_name,
                    "id": it.id,
                    "repeat": r,
                    "gold_is_faithful": it.gold_is_faithful,
                    "faithfulness_score": report.faithfulness_score,
                    "n_claims": report.n_claims,
                    "n_contradicted": report.n_contradicted,
                    "n_refused": report.n_refused,
                    "predicted_unfaithful": {
                        _tau_key(t): predicted_unfaithful(report, t) for t in taus
                    },
                }
            )
        tau_prf: dict[float, dict] = {}
        for t in taus:
            preds = [predicted_unfaithful(rep, t) for rep in reports]
            tau_prf[t] = binary_prf(gold_unfaithful, preds)
        per_repeat.append(tau_prf)

    return Tier2SliceResult(
        slice_name=slice_name,
        n_items=len(items),
        repeats=repeats,
        aggregate=_aggregate_tier2(per_repeat, taus),
        refusal_affected=refusal_affected,
        item_lines=item_lines,
    )


def _aggregate_tier2(per_repeat: list[dict[float, dict]], taus: tuple[float, ...]) -> dict:
    """Mean ± spread per τ for precision / recall / F1 / accuracy across repeats."""
    out: dict[str, dict] = {}
    for t in taus:
        out[_tau_key(t)] = {
            metric: _mean_spread([rep[t][metric] for rep in per_repeat])
            for metric in ("precision", "recall", "f1", "accuracy")
        }
    return out


def _tier2_slice_summary(result: Tier2SliceResult) -> dict:
    """A Tier-2 slice headline: the τ sweep (positive = unfaithful), with the operating point."""
    return {
        "slice": result.slice_name,
        "n": result.n_items,
        "repeats": result.repeats,
        "operating_point_tau": _tau_key(TAU_SWEEP[0]),
        "tau_sweep": result.aggregate,
    }


def _tau_key(tau: float) -> str:
    """Stable string key for a τ value (JSON object keys must be strings)."""
    return f"{tau:g}"


# --------------------------------------------------------------------------- #
# Run orchestration + persistence
# --------------------------------------------------------------------------- #


def _timestamp() -> str:
    """UTC ISO-ish timestamp for the run header + default filename (top-level OS clock)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run(
    *,
    tiers: list[str],
    n: int,
    repeats: int,
    slice_which: str,
    out_path: str | Path,
    provider: LLMProvider,
    mock_mode: bool,
    taus: tuple[float, ...] = TAU_SWEEP,
    datasets_dir: Path = DATASETS_DIR,
    datasets: Optional[dict[str, LoadedDataset]] = None,
    write: bool = True,
) -> tuple[dict, list[dict], Path]:
    """Run the requested tiers and persist a ``runs/*.jsonl`` (header + items + summary).

    Returns ``(summary, lines, out_path)``. ``datasets`` lets a caller inject
    in-memory :class:`LoadedDataset` objects (the test seam) instead of reading the
    YAML files. Tier failures from missing keys propagate as ``RuntimeError`` for
    the CLI to surface cleanly.
    """
    datasets = datasets or {}
    ts = _timestamp()
    header = {
        "kind": "run_header",
        "prompt_version": PROMPT_VERSION,
        "decompose_model": DECOMPOSE_MODEL,
        "ground_model": GROUND_MODEL,
        "n": n,
        "repeats": repeats,
        "tiers": tiers,
        "slice": slice_which,
        "mock_mode": mock_mode,
        "taus": [_tau_key(t) for t in taus],
        "ts": ts,
    }
    lines: list[dict] = [header]
    summary: dict = {
        "kind": "summary",
        "ts": ts,
        "n": n,
        "repeats": repeats,
        "mock_mode": mock_mode,
        "tiers": {},
    }
    refusal_affected = 0
    skipped = 0

    if "1" in tiers:
        ds = datasets.get("1") or load_dataset(datasets_dir / "tier1_claims.yaml")
        skipped += _log_skipped(ds, "tier1")
        tier1_summary: dict = {}
        for sname, sitems in split_slice(ds.items, slice_which).items():
            res = run_tier1(sitems, n=n, repeats=repeats, provider=provider, slice_name=sname)
            lines.extend(res.item_lines)
            refusal_affected += res.refusal_affected
            tier1_summary[sname] = _tier1_slice_summary(res, frozen=(sname == "frozen"))
        summary["tiers"]["1"] = tier1_summary

    if "2" in tiers:
        ds = datasets.get("2") or load_dataset(datasets_dir / "tier2_answers.yaml")
        skipped += _log_skipped(ds, "tier2")
        tier2_summary: dict = {}
        for sname, sitems in split_slice(ds.items, slice_which).items():
            res = run_tier2(
                sitems, n=n, repeats=repeats, provider=provider, taus=taus, slice_name=sname
            )
            lines.extend(res.item_lines)
            refusal_affected += res.refusal_affected
            tier2_summary[sname] = _tier2_slice_summary(res)
        summary["tiers"]["2"] = tier2_summary

    # Honesty surfacing (spec §7/§11): refusal-driven drops must be distinguishable.
    summary["refusal_affected"] = refusal_affected
    summary["skipped"] = skipped
    lines.append(summary)

    out_path = Path(out_path)
    if write:
        _write_jsonl(out_path, lines)
    return summary, lines, out_path


def _log_skipped(ds: LoadedDataset, label: str) -> int:
    """Warn (never crash) for each malformed-and-skipped dataset item; return the count."""
    for raw, reason in ds.skipped:
        ident = raw.get("id", "<no id>") if isinstance(raw, dict) else "<non-mapping>"
        logger.warning("%s: skipping malformed item %r — %s", label, ident, reason)
    return len(ds.skipped)


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    """Write one JSON object per line (the reproducibility log, spec §11)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _fmt(ms: dict[str, float]) -> str:
    """ASCII mean ± spread, e.g. ``0.694 +/- 0.012 [0.68, 0.71]`` (robust on Windows)."""
    return (
        f"{ms['mean']:.3f} +/- {ms['stdev']:.3f} "
        f"[{ms['min']:.3f}, {ms['max']:.3f}]"
    )


def _print_summary(summary: dict) -> None:
    """Print the headline numbers (held-in macro-F1 + κ, frozen accuracy + κ, τ sweep)."""
    print("=" * 64)
    print(
        f"GroundCheck meta-eval  (n={summary['n']}, R={summary['repeats']}, "
        f"mock_mode={summary['mock_mode']})"
    )
    tiers = summary.get("tiers", {})

    if "1" in tiers:
        print("\nTier 1 -- grounding-judge accuracy (3-class)")
        for sname, s in tiers["1"].items():
            print(f"  [{sname}] n={s['n']}")
            if "macro_f1" in s:  # held-in
                print(f"    macro-F1 : {_fmt(s['macro_f1'])}")
                print(f"    accuracy : {_fmt(s['accuracy'])}")
                print(f"    kappa    : {_fmt(s['kappa'])}")
                for c in LABELS:
                    print(f"    F1[{c:<16}]: {_fmt(s['per_class'][c]['f1'])}")
            else:  # frozen — accuracy + kappa only
                print(f"    accuracy : {_fmt(s['accuracy'])}")
                print(f"    kappa    : {_fmt(s['kappa'])}")
                print(f"    note     : {s['caption']}")

    if "2" in tiers:
        print("\nTier 2 -- end-to-end detection (positive = unfaithful)")
        for sname, s in tiers["2"].items():
            print(f"  [{sname}] n={s['n']}  operating point tau={s['operating_point_tau']}")
            for tau, prf in s["tau_sweep"].items():
                print(
                    f"    tau={tau}: P {_fmt(prf['precision'])}  "
                    f"R {_fmt(prf['recall'])}"
                )

    if summary.get("refusal_affected"):
        print(f"\n[!] refusal-affected checks: {summary['refusal_affected']} (spec 7/11)")
    if summary.get("skipped"):
        print(f"[!] malformed items skipped: {summary['skipped']}")
    print("=" * 64)


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m eval.run`` entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="GroundCheck two-tier meta-eval harness (spec §11).",
    )
    parser.add_argument("--tier", choices=["1", "2", "all"], default="all")
    parser.add_argument("-n", type=int, default=DEFAULT_N_RUNS, help="grounding runs per claim")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="R: whole-tier repeats")
    parser.add_argument("--quick", action="store_true", help="fast iteration: n=1, R=1")
    parser.add_argument(
        "--slice",
        choices=["held-in", "frozen", "all"],
        default="all",
        dest="slice_which",
        help="which slice(s) to report (default: both, separately)",
    )
    parser.add_argument("--out", default=None, help="JSONL output path (default: runs/<ts>.jsonl)")
    args = parser.parse_args(argv)

    n = 1 if args.quick else args.n
    repeats = 1 if args.quick else args.repeats
    tiers = ["1", "2"] if args.tier == "all" else [args.tier]
    out_path = Path(args.out) if args.out else RUNS_DIR / f"{_timestamp()}.jsonl"
    mock_mode = (os.getenv("GROUNDCHECK_LLM", "").strip().lower() == "mock")

    provider = get_provider()
    try:
        summary, _lines, out_path = run(
            tiers=tiers,
            n=n,
            repeats=repeats,
            slice_which=args.slice_which,
            out_path=out_path,
            provider=provider,
            mock_mode=mock_mode,
        )
    except RuntimeError as exc:  # clean missing-key / provider message — no traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    print(f"\nrun persisted to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
