"""Tests for the pure majority/confidence helpers (Split 04, no key).

These are the determinism heart of the firewall (spec §9): the severity tie-break
must bias a split vote toward *flagging*, never silently certify SUPPORTED.
"""

from __future__ import annotations

import pytest

from groundcheck.ground import confidence, majority_label

S = "SUPPORTED"
N = "NOT_ENOUGH_INFO"
C = "CONTRADICTED"


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([S, S, S], S),
        ([S, S, N], S),
        ([N, N, C], N),
        ([S, N, C], C),  # 1-1-1 → most severe present
        ([S, N], N),  # even tie → more severe of the two
        ([S, C], C),  # even tie → more severe of the two
        ([N, N, S, S], N),  # 2-2 tie → more severe of the tied
        ([C], C),
        # extra: a clear contradicted majority and an all-NEI vote
        ([C, C, S], C),
        ([N, N, N], N),
    ],
)
def test_majority_label_table(labels, expected):
    assert majority_label(labels) == expected


def test_majority_label_is_not_most_common_order_sensitive():
    # Counter.most_common is insertion-order-arbitrary on ties; the severity rule is
    # not. Both orderings of a 1-1-1 vote must resolve to CONTRADICTED.
    assert majority_label([S, N, C]) == C
    assert majority_label([C, N, S]) == C
    assert majority_label([N, C, S]) == C


def test_confidence():
    assert confidence({"SUPPORTED": 3}, 3) == 1.0
    assert confidence({"NOT_ENOUGH_INFO": 2, "SUPPORTED": 1}, 3) == pytest.approx(0.6667, abs=1e-3)
    assert confidence({"CONTRADICTED": 1}, 1) == 1.0


def test_confidence_degenerate_inputs_do_not_crash():
    assert confidence({}, 3) == 0.0
    assert confidence({"SUPPORTED": 1}, 0) == 0.0
