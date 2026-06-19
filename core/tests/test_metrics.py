"""Tests for Split 06 — dependency-free meta-eval metrics (spec §11). No key needed.

Every expected number is hand-worked in the split brief or in an inline comment; we
never let the implementation define the expectation. ``pytest.approx`` absorbs float
noise on the rates.
"""

from __future__ import annotations

import pytest

from groundcheck.metrics import (
    accuracy,
    binary_prf,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    per_class_prf,
    tier1_report,
)

# Tier-1 label set and the 10-item worked fixture (SPLIT-06 §"Hand-worked fixture").
LABELS = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
S, C, N = LABELS
GOLD = [S, S, S, C, C, C, N, N, N, N]
PRED = [S, S, C, C, C, N, N, N, N, S]


# --------------------------------------------------------------------------- #
# Tier 1 — the worked 10-item fixture
# --------------------------------------------------------------------------- #


def test_tier1_worked_fixture():
    """All six headline numbers match the brief's hand-worked table."""
    report = tier1_report(GOLD, PRED, LABELS)
    assert report["accuracy"] == pytest.approx(0.7)  # 7/10 correct
    assert report["macro_f1"] == pytest.approx(0.694444, abs=1e-6)
    assert report["kappa"] == pytest.approx(0.545455, abs=1e-6)
    assert report["n"] == 10

    pc = report["per_class"]
    for label, expected in (
        (S, 0.666667),
        (C, 0.666667),
        (N, 0.75),
    ):
        assert pc[label]["precision"] == pytest.approx(expected, abs=1e-6)
        assert pc[label]["recall"] == pytest.approx(expected, abs=1e-6)
        assert pc[label]["f1"] == pytest.approx(expected, abs=1e-6)


def test_confusion_matrix_shape():
    """The confusion matrix equals the brief's table exactly (integer counts)."""
    cm = confusion_matrix(GOLD, PRED, LABELS)
    assert cm == {
        S: {S: 2, C: 1, N: 0},
        C: {S: 0, C: 2, N: 1},
        N: {S: 1, C: 0, N: 3},
    }


def test_per_class_support_is_gold_count():
    """Support is the number of gold items of each class (3 / 3 / 4), as ints."""
    pc = per_class_prf(GOLD, PRED, LABELS)
    assert pc[S]["support"] == 3
    assert pc[C]["support"] == 3
    assert pc[N]["support"] == 4


def test_macro_f1_is_unweighted_mean():
    """macro-F1 is the plain mean of the three per-class F1s, not support-weighted."""
    pc = per_class_prf(GOLD, PRED, LABELS)
    expected = (pc[S]["f1"] + pc[C]["f1"] + pc[N]["f1"]) / 3
    assert macro_f1(GOLD, PRED, LABELS) == pytest.approx(expected)
    # A support-weighted mean would weight N (support 4) more heavily; confirm we did
    # NOT do that. Weighted = (3*F1_S + 3*F1_C + 4*F1_N)/10.
    weighted = (3 * pc[S]["f1"] + 3 * pc[C]["f1"] + 4 * pc[N]["f1"]) / 10
    assert macro_f1(GOLD, PRED, LABELS) != pytest.approx(weighted)


def test_kappa_formula():
    """κ for the 10-item fixture, plus a second independently hand-worked case."""
    # Fixture: po=0.7; marginals pred S3/C3/N4, gold S3/C3/N4, N=10.
    # pe = .3*.3 + .3*.3 + .4*.4 = 0.09 + 0.09 + 0.16 = 0.34.
    # κ = (0.7 - 0.34)/(1 - 0.34) = 0.36/0.66 = 0.545454...
    assert cohen_kappa(GOLD, PRED, LABELS) == pytest.approx(0.545455, abs=1e-6)

    # Second case (2-class, 6 items), worked by hand:
    #   gold: A A A B B B   pred: A A B B B A   labels [A, B]
    #   confusion: A->{A:2,B:1}, B->{B:2,A:1}; correct = 4 -> po = 4/6 = 0.6667
    #   marginals: pred A=3, pred B=3, gold A=3, gold B=3, N=6
    #   pe = (3/6)(3/6) + (3/6)(3/6) = 0.25 + 0.25 = 0.5
    #   κ = (0.6667 - 0.5)/(1 - 0.5) = 0.1667/0.5 = 0.3333
    g2 = ["A", "A", "A", "B", "B", "B"]
    p2 = ["A", "A", "B", "B", "B", "A"]
    assert cohen_kappa(g2, p2, ["A", "B"]) == pytest.approx(1 / 3, abs=1e-6)


# --------------------------------------------------------------------------- #
# Degenerate / edge cases
# --------------------------------------------------------------------------- #


def test_perfect_agreement():
    """gold == pred (mix of classes) → accuracy / every F1 / macro-F1 / κ all 1.0."""
    gold = [S, S, C, C, N, N]
    pred = list(gold)
    report = tier1_report(gold, pred, LABELS)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["kappa"] == pytest.approx(1.0)
    for label in LABELS:
        assert report["per_class"][label]["f1"] == pytest.approx(1.0)


def test_degenerate_single_class():
    """All one class on both sides → accuracy 1.0, κ 1.0 (pe==1 branch; no crash)."""
    gold = [S, S, S, S]
    pred = [S, S, S, S]
    assert accuracy(gold, pred) == 1.0
    # pe = (4/4)(4/4) = 1.0 → degenerate branch → po==1.0 → 1.0
    assert cohen_kappa(gold, pred, LABELS) == 1.0


def test_zero_division_precision():
    """A class present in gold but never predicted → its P/R/F1 = 0.0, no exception."""
    # gold has C's, but pred never says C; pred also never matches the gold C's.
    gold = [S, S, C, C]
    pred = [S, S, S, N]  # C predicted zero times; the two gold C's are missed
    pc = per_class_prf(gold, pred, LABELS)
    assert pc[C]["precision"] == 0.0  # 0 predicted → 0/0 → 0.0
    assert pc[C]["recall"] == 0.0  # 0 of 2 gold C's recovered
    assert pc[C]["f1"] == 0.0
    assert pc[C]["support"] == 2  # support still reflects the gold count


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([S, S], [S])
    with pytest.raises(ValueError):
        confusion_matrix([S, S], [S], LABELS)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        confusion_matrix([], [], LABELS)
    with pytest.raises(ValueError):
        binary_prf([], [])


def test_unknown_label_raises():
    """A value outside `labels` → ValueError from confusion_matrix (and callers)."""
    with pytest.raises(ValueError):
        confusion_matrix([S, "MAYBE"], [S, S], LABELS)
    with pytest.raises(ValueError):
        confusion_matrix([S, S], [S, "MAYBE"], LABELS)
    with pytest.raises(ValueError):
        cohen_kappa([S, "MAYBE"], [S, S], LABELS)


# --------------------------------------------------------------------------- #
# Tier 2 — binary detection metrics
# --------------------------------------------------------------------------- #


def test_binary_prf_worked_fixture():
    """The 5-item binary fixture → exact counts + rates (SPLIT-06 brief)."""
    # gold: T T T F F   pred: T F T F T  → TP=2, FP=1, FN=1, TN=1
    gold = [True, True, True, False, False]
    pred = [True, False, True, False, True]
    r = binary_prf(gold, pred)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 1)
    assert r["precision"] == pytest.approx(0.666667, abs=1e-6)
    assert r["recall"] == pytest.approx(0.666667, abs=1e-6)
    assert r["f1"] == pytest.approx(0.666667, abs=1e-6)
    assert r["accuracy"] == pytest.approx(0.6)


def test_binary_prf_all_negative():
    """No positives anywhere → P/R/F1 = 0.0 (0/0 convention), accuracy 1.0, no crash."""
    gold = [False, False, False]
    pred = [False, False, False]
    r = binary_prf(gold, pred)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (0, 0, 0, 3)
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0
    assert r["accuracy"] == 1.0


def test_binary_prf_perfect():
    """Perfect detection → all rates 1.0."""
    gold = [True, False, True, False]
    pred = [True, False, True, False]
    r = binary_prf(gold, pred)
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0
    assert r["accuracy"] == 1.0


def test_binary_prf_rejects_non_bool():
    """Non-bool values (e.g. 1/0 ints) → ValueError, not a silent truthy coercion."""
    with pytest.raises(ValueError):
        binary_prf([1, 0, 1], [True, False, True])
    with pytest.raises(ValueError):
        binary_prf([True, False], [1, 0])


# --------------------------------------------------------------------------- #
# Order-independence & determinism
# --------------------------------------------------------------------------- #


def test_metrics_are_order_independent():
    """Permuting (gold, pred) pairs together leaves every metric unchanged."""
    pairs = list(zip(GOLD, PRED))
    shuffled = pairs[3:] + pairs[:3]  # deterministic rotation, no RNG
    g2, p2 = map(list, zip(*shuffled))
    assert tier1_report(g2, p2, LABELS) == tier1_report(GOLD, PRED, LABELS)


def test_public_api_exports():
    """The seven metric functions are importable straight off `groundcheck`."""
    import groundcheck

    for name in (
        "confusion_matrix",
        "per_class_prf",
        "macro_f1",
        "accuracy",
        "cohen_kappa",
        "tier1_report",
        "binary_prf",
    ):
        assert hasattr(groundcheck, name)
