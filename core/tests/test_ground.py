"""Tests for Split 04 — N-run grounding (no key; one optional @api smoke).

Every no-key test injects a provider (a recording stub or :class:`MockProvider`) so
the suite never touches the network.
"""

from __future__ import annotations

import pytest

from groundcheck.config import GROUND_MODEL
from groundcheck.ground import (
    GroundOutcome,
    REFUSAL_RATIONALE,
    ground,
    ground_once,
)
from groundcheck.llm import MockProvider, ParseResult, Usage, compute_cost
from groundcheck.models import GroundingVerdict
from groundcheck.worked_example import (
    REFUSAL_TRIGGER,
    WORKED_EXAMPLE_CLAIMS,
    WORKED_EXAMPLE_SOURCE,
    WORKED_EXAMPLE_VERDICTS,
)

# §5 expected verdicts: claims 1,3,4,7,8 → SUPPORTED; 2,5,6 → NOT_ENOUGH_INFO.
EXPECTED_LABELS = [
    "SUPPORTED",
    "NOT_ENOUGH_INFO",
    "SUPPORTED",
    "SUPPORTED",
    "NOT_ENOUGH_INFO",
    "NOT_ENOUGH_INFO",
    "SUPPORTED",
    "SUPPORTED",
]


# --------------------------------------------------------------------------- #
# Test doubles (dependency injection — no network)
# --------------------------------------------------------------------------- #


class _RecordingProvider:
    """Records every parse() call; returns a fixed ParseResult (default: a verdict)."""

    def __init__(self, result: ParseResult | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result

    def parse(self, **kwargs) -> ParseResult:
        self.calls.append(kwargs)
        if self._result is not None:
            return self._result
        return ParseResult(
            parsed=GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"),
            usage=Usage(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            refused=False,
            truncated=False,
        )


# --------------------------------------------------------------------------- #
# ground_once: cache shape, refusal mapping, degradation
# --------------------------------------------------------------------------- #


def test_ground_once_sends_cached_source_block():
    spy = _RecordingProvider()
    ground_once("SRC TEXT", "CLAIM TEXT", provider=spy)
    assert len(spy.calls) == 1
    blocks = spy.calls[0]["user_blocks"]
    # First block = the cached SOURCE prefix; a later block carries the CLAIM.
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["text"].startswith("SOURCE:")
    assert "SRC TEXT" in blocks[0]["text"]
    assert any("CLAIM:" in b["text"] for b in blocks[1:])
    assert "cache_control" not in blocks[1]  # the CLAIM block is not cached
    # Routed to Opus with the ground rubric and the small ground budget.
    assert spy.calls[0]["model"] == GROUND_MODEL
    assert spy.calls[0]["output_model"] is GroundingVerdict


def test_ground_once_refusal_maps_to_nei():
    refusal = ParseResult(
        parsed=None, usage=Usage(input_tokens=5), stop_reason="refusal", refused=True, truncated=False
    )
    verdict, refused, usage = ground_once("s", "c", provider=_RecordingProvider(refusal))
    assert verdict.label == "NOT_ENOUGH_INFO"
    assert verdict.supporting_span == ""
    assert verdict.rationale == REFUSAL_RATIONALE == "model declined to judge"
    assert refused is True
    assert usage == Usage(input_tokens=5)


def test_ground_once_unparseable_degrades_to_nei_not_refused():
    # A non-refusal that returns no verdict (e.g. truncated) must not crash and must
    # not be mislabelled a refusal.
    broken = ParseResult(
        parsed=None, usage=Usage(), stop_reason="max_tokens", refused=False, truncated=True
    )
    verdict, refused, _ = ground_once("s", "c", provider=_RecordingProvider(broken))
    assert verdict.label == "NOT_ENOUGH_INFO"
    assert refused is False


# --------------------------------------------------------------------------- #
# ground: the §5 worked-example verdicts (5 SUPPORTED + 3 NEI)
# --------------------------------------------------------------------------- #


def test_ground_worked_example_verdicts():
    provider = MockProvider()
    labels = [
        ground(WORKED_EXAMPLE_SOURCE, c.claim, n=3, provider=provider).label
        for c in WORKED_EXAMPLE_CLAIMS
    ]
    assert labels == EXPECTED_LABELS
    assert labels.count("SUPPORTED") == 5
    assert labels.count("NOT_ENOUGH_INFO") == 3


def test_ground_representative_span_from_winning_run():
    provider = MockProvider()
    # Claim 1 wins SUPPORTED → span/rationale come from the SUPPORTED run.
    out1 = ground(WORKED_EXAMPLE_SOURCE, WORKED_EXAMPLE_CLAIMS[0].claim, n=3, provider=provider)
    assert out1.label == "SUPPORTED"
    assert out1.supporting_span == "usually has no symptoms"
    assert out1.rationale == "The source states hypertension usually has no symptoms."
    # Claim 5 wins NEI (split vote) → span forced to "", rationale from the NEI run,
    # never the spurious minority SUPPORTED.
    out5 = ground(WORKED_EXAMPLE_SOURCE, WORKED_EXAMPLE_CLAIMS[4].claim, n=3, provider=provider)
    assert out5.label == "NOT_ENOUGH_INFO"
    assert out5.supporting_span == ""
    assert out5.rationale == "The source gives no magnitude for reducing salt."


# --------------------------------------------------------------------------- #
# Split votes: claims 2 & 5 vote 2·1 / confidence 0.67 (load-bearing for 09/11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("claim_index", [1, 4])  # claim 2 and claim 5
def test_ground_split_vote_confidence(claim_index):
    out = ground(
        WORKED_EXAMPLE_SOURCE,
        WORKED_EXAMPLE_CLAIMS[claim_index].claim,
        n=3,
        provider=MockProvider(),
    )
    assert out.label == "NOT_ENOUGH_INFO"
    assert out.votes == {"NOT_ENOUGH_INFO": 2, "SUPPORTED": 1}
    assert out.confidence == pytest.approx(0.6667, abs=1e-3)
    assert out.refused is False


def test_ground_clean_vote_confidence_is_one():
    # Claim 6 is a clean 3·0 NEI → confidence 1.0 (the dc-html "3 · 0" data).
    out = ground(WORKED_EXAMPLE_SOURCE, WORKED_EXAMPLE_CLAIMS[5].claim, n=3, provider=MockProvider())
    assert out.label == "NOT_ENOUGH_INFO"
    assert out.votes == {"NOT_ENOUGH_INFO": 3}
    assert out.confidence == 1.0


def test_split_vote_reproducible_across_calls():
    # Re-running ground() on the same provider must reproduce the vote spread (the mock
    # sequence index cycles cleanly at n == sequence length).
    provider = MockProvider()
    claim = WORKED_EXAMPLE_CLAIMS[4].claim
    a = ground(WORKED_EXAMPLE_SOURCE, claim, n=3, provider=provider)
    b = ground(WORKED_EXAMPLE_SOURCE, claim, n=3, provider=provider)
    assert a == b


def test_worked_example_n_low_confidence_is_two():
    # Across the 8 claims, exactly claims 2 and 5 are below 1.0 → the UI "· 2 to review".
    provider = MockProvider()
    confs = [
        ground(WORKED_EXAMPLE_SOURCE, c.claim, n=3, provider=provider).confidence
        for c in WORKED_EXAMPLE_CLAIMS
    ]
    assert sum(1 for c in confs if c < 1.0) == 2


# --------------------------------------------------------------------------- #
# Refusal → NEI end to end (uses the pinned, seeded REFUSAL_TRIGGER)
# --------------------------------------------------------------------------- #


def test_ground_refusal_endtoend():
    out = ground(WORKED_EXAMPLE_SOURCE, REFUSAL_TRIGGER, n=3, provider=MockProvider())
    assert out.label == "NOT_ENOUGH_INFO"
    assert out.refused is True
    assert out.n_refused_runs == 3
    assert out.confidence == 1.0
    assert out.rationale == REFUSAL_RATIONALE


class _PartialRefusalProvider:
    """Refuses on the first call, returns SUPPORTED afterward (per provider instance)."""

    def __init__(self) -> None:
        self._n = 0

    def parse(self, **kwargs) -> ParseResult:
        self._n += 1
        if self._n == 1:
            return ParseResult(
                parsed=None, usage=Usage(), stop_reason="refusal", refused=True, truncated=False
            )
        return ParseResult(
            parsed=GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"),
            usage=Usage(),
            stop_reason="end_turn",
            refused=False,
            truncated=False,
        )


def test_ground_partial_refusal_counts_but_does_not_flip_majority():
    # One refusal (→ NEI) + two SUPPORTED → majority SUPPORTED, but the refusal is still
    # surfaced via refused / n_refused_runs (spec §7 honesty: refusal-driven loss visible).
    out = ground("s", "c", n=3, provider=_PartialRefusalProvider())
    assert out.label == "SUPPORTED"
    assert out.votes == {"NOT_ENOUGH_INFO": 1, "SUPPORTED": 2}
    assert out.refused is True
    assert out.n_refused_runs == 1


# --------------------------------------------------------------------------- #
# Cost sums across runs; empty source / no crash
# --------------------------------------------------------------------------- #


def test_ground_cost_sums_runs():
    usage = Usage(input_tokens=100, cache_read_input_tokens=20, output_tokens=30)
    result = ParseResult(
        parsed=GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"),
        usage=usage,
        stop_reason="end_turn",
        refused=False,
        truncated=False,
    )
    out = ground("s", "c", n=3, provider=_RecordingProvider(result))
    assert out.cost_usd == pytest.approx(3 * compute_cost(GROUND_MODEL, usage))
    assert out.cost_usd > 0.0


def test_ground_empty_source_does_not_crash():
    out = ground("", "some unregistered claim", n=2, provider=MockProvider())
    assert isinstance(out, GroundOutcome)
    assert out.label == "NOT_ENOUGH_INFO"  # mock default for an unknown claim
    assert out.refused is False


def test_ground_is_sequential_n_calls():
    spy = _RecordingProvider()
    ground("s", "c", n=4, provider=spy)
    assert len(spy.calls) == 4  # exactly N calls, in order, no threads here


# --------------------------------------------------------------------------- #
# Fixture invariants: SUPPORTED spans are verbatim source substrings; NEI spans empty
# --------------------------------------------------------------------------- #


def test_worked_example_verdict_spans_are_grounded():
    for value in WORKED_EXAMPLE_VERDICTS.values():
        verdicts = value if isinstance(value, list) else [value]
        for v in verdicts:
            if v.label == "SUPPORTED" and v.supporting_span:
                assert v.supporting_span in WORKED_EXAMPLE_SOURCE
            if v.label == "NOT_ENOUGH_INFO":
                assert v.supporting_span == ""


# --------------------------------------------------------------------------- #
# Optional live smoke (needs a real key; skipped otherwise) — the demo headline
# --------------------------------------------------------------------------- #


@pytest.mark.api
def test_real_ground_smoke(real_provider):
    # Claim 3 is stated verbatim → SUPPORTED; claim 5's "exactly 25%" is fabricated →
    # NOT_ENOUGH_INFO (the demo's headline). Distributional, so assert these two only.
    out3 = ground(WORKED_EXAMPLE_SOURCE, WORKED_EXAMPLE_CLAIMS[2].claim, n=3, provider=real_provider)
    out5 = ground(WORKED_EXAMPLE_SOURCE, WORKED_EXAMPLE_CLAIMS[4].claim, n=3, provider=real_provider)
    assert out3.label == "SUPPORTED"
    assert out5.label == "NOT_ENOUGH_INFO"
    assert out3.cost_usd > 0.0 and out5.cost_usd > 0.0
