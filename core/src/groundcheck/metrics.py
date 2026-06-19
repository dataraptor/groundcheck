"""Dependency-free classification metrics for the two-tier meta-eval (spec §11, §20-D).

Pure stdlib — no third-party numerical or ML libraries (decision §20-D: hand-roll
one κ rather than take a heavy dependency for it). Every function here is **pure**
(no I/O, no global state) and takes plain Python lists, so the whole module is a
no-key Tier-1 unit test target (spec §12) and is reused by the eval harness in
Split 07.

The two tiers it serves:

* **Tier 1 — grounding-judge accuracy** (3-class): per-class precision / recall /
  F1, macro-F1 (the headline), accuracy, and Cohen's κ (detector-majority vs gold).
* **Tier 2 — answer-level detection** (binary): precision / recall / F1 on the
  ``True`` = "unfaithful" positive class, plus accuracy, for the τ-sweep.

Conventions (deliberate choices Split 07 relies on — do not "fix" them):

* **Length mismatch / empty input → ``ValueError``** from every entry point.
* **Unknown label** (a value not in ``labels``) → ``ValueError`` from
  :func:`confusion_matrix` and anything built on it. :func:`binary_prf` accepts
  only ``bool`` values; any non-bool → ``ValueError``.
* **0/0 → 0.0** for precision / recall / F1. A class that is never *predicted* has
  precision 0.0; a class never present in *gold* has recall 0.0. This keeps the
  frozen ~9-item slice (≈3/class) from crashing when a class draws zero predictions
  (spec §11 "Size honesty"), and is a documented choice, not an accident.
* **κ degenerate case** (``pe == 1`` — all gold *and* all pred are a single class):
  return ``1.0`` if the labelings agree perfectly, else ``0.0``. No ZeroDivisionError.
* **macro-F1 is the *unweighted* class mean** (spec §11 headline) — every class
  counts equally regardless of support, *not* a support-weighted average. This is
  the metric that punishes ignoring a rare class.

The κ formula is fixed by §11 and implemented here directly (not imported):
``po = accuracy``; ``pe = Σ_c (pred_c/N)(gold_c/N)``; ``κ = (po − pe)/(1 − pe)``.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _validate_pair(gold: list, pred: list) -> None:
    """Raise ``ValueError`` if ``gold``/``pred`` differ in length or are empty."""
    if len(gold) != len(pred):
        raise ValueError(
            f"length mismatch: len(gold)={len(gold)} != len(pred)={len(pred)}"
        )
    if not gold:
        raise ValueError("empty input: gold and pred must be non-empty")


def _safe_div(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or ``0.0`` when the denominator is 0 (0/0 → 0.0)."""
    return numerator / denominator if denominator else 0.0


# --------------------------------------------------------------------------- #
# Tier 1 — 3-class metrics
# --------------------------------------------------------------------------- #


def confusion_matrix(
    gold: list[str], pred: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    """Confusion matrix as nested dicts: rows = gold, cols = pred.

    ``cm[g][p]`` is the count of items whose gold label is ``g`` and predicted
    label is ``p``. Every (``g``, ``p``) cell over ``labels`` is present (0 where
    no items fall in it), so callers can index any label pair without a ``KeyError``.

    Raises ``ValueError`` if ``len(gold) != len(pred)``, if either is empty, or if
    any value in ``gold``/``pred`` is not in ``labels``.
    """
    _validate_pair(gold, pred)
    known = set(labels)
    cm: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g not in known:
            raise ValueError(f"unknown gold label {g!r} (not in labels={labels})")
        if p not in known:
            raise ValueError(f"unknown pred label {p!r} (not in labels={labels})")
        cm[g][p] += 1
    return cm


def per_class_prf(
    gold: list[str], pred: list[str], labels: list[str]
) -> dict[str, dict[str, float]]:
    """Per-class precision / recall / F1 / support, keyed by label.

    For each class ``c``: ``precision = TP/(TP+FP)``, ``recall = TP/(TP+FN)``,
    ``f1 = 2PR/(P+R)``, and ``support`` = number of gold items of class ``c``.
    When a denominator is 0 the corresponding metric is ``0.0`` (module convention).
    """
    cm = confusion_matrix(gold, pred, labels)
    report: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = cm[c][c]
        gold_c = sum(cm[c][p] for p in labels)  # row sum = # gold of class c
        pred_c = sum(cm[g][c] for g in labels)  # col sum = # predicted as c
        fp = pred_c - tp
        fn = gold_c - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": gold_c,
        }
    return report


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    """Unweighted mean of per-class F1 over ``labels`` (every class weighted equally)."""
    prf = per_class_prf(gold, pred, labels)
    return sum(prf[c]["f1"] for c in labels) / len(labels)


def accuracy(gold: list, pred: list) -> float:
    """Fraction of items where ``gold == pred``. Raises on length mismatch / empty input."""
    _validate_pair(gold, pred)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return correct / len(gold)


def cohen_kappa(gold: list[str], pred: list[str], labels: list[str]) -> float:
    """Cohen's κ exactly per spec §11.

    ``po = accuracy``; ``pe = Σ_c (pred_c/N)(gold_c/N)`` over the marginal counts;
    ``κ = (po − pe)/(1 − pe)``. When ``1 − pe == 0`` (degenerate: all gold and all
    pred collapse to one class) return ``1.0`` if ``po == 1.0`` else ``0.0``.
    """
    cm = confusion_matrix(gold, pred, labels)  # validates inputs/labels
    n = len(gold)
    po = accuracy(gold, pred)
    pe = 0.0
    for c in labels:
        gold_c = sum(cm[c][p] for p in labels)
        pred_c = sum(cm[g][c] for g in labels)
        pe += (pred_c / n) * (gold_c / n)
    if 1 - pe == 0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def tier1_report(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    """Bundle every Tier-1 number the leaderboard needs (spec §11).

    Returns ``{"per_class", "macro_f1", "accuracy", "kappa", "n"}`` — per-class
    P/R/F1, the macro-F1 headline, accuracy, Cohen's κ, and the item count.
    """
    return {
        "per_class": per_class_prf(gold, pred, labels),
        "macro_f1": macro_f1(gold, pred, labels),
        "accuracy": accuracy(gold, pred),
        "kappa": cohen_kappa(gold, pred, labels),
        "n": len(gold),
    }


# --------------------------------------------------------------------------- #
# Tier 2 — binary detection metrics (positive class = True = "unfaithful")
# --------------------------------------------------------------------------- #


def binary_prf(gold: list[bool], pred: list[bool]) -> dict[str, float]:
    """Binary precision / recall / F1 / accuracy for the τ-sweep (spec §11 Tier 2).

    The positive class is ``True`` ("unfaithful"). Returns the four rates
    (``precision``, ``recall``, ``f1``, ``accuracy``) plus the raw counts
    (``tp``, ``fp``, ``fn``, ``tn`` as ints). 0/0 denominators → ``0.0`` (module
    convention). Accepts only ``bool`` values; any non-bool raises ``ValueError``.
    """
    _validate_pair(gold, pred)
    for value in (*gold, *pred):
        if not isinstance(value, bool):
            raise ValueError(f"binary_prf expects bool values; got {value!r}")

    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
