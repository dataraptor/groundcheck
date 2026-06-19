"""Tests for Split 05 — the orchestrator ``check()`` (no key; one optional @api smoke).

Every no-key test injects a provider (a stub or :class:`MockProvider`) so the suite
never touches the network.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from groundcheck import check
from groundcheck.config import (
    DECOMPOSE_MODEL,
    GROUND_MODEL,
    SOURCE_CACHE_FLOOR_TOKENS,
    VERDICT_COLORS,
)
from groundcheck.llm import REFUSAL, MockProvider, ParseResult, Usage, compute_cost
from groundcheck.models import DecomposedClaim, Decomposition, GroundingVerdict
from groundcheck.pipeline import _should_warm_cache_first

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
AMBER = VERDICT_COLORS["NOT_ENOUGH_INFO"]


def _load(name: str) -> tuple[str, str]:
    data = json.loads((_EXAMPLES / name).read_text(encoding="utf-8"))
    return data["source"], data["answer"]


# --------------------------------------------------------------------------- #
# N/A path: 0 claims (empty / whitespace answer)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
def test_zero_claims_is_na(answer):
    report = check("a source", answer, n=3, provider=MockProvider())
    assert report.faithfulness_score is None
    assert report.n_claims == 0
    assert report.n_supported == 0
    assert report.cost_usd == 0.0  # decompose short-circuits with no spend
    assert "<span" not in report.highlighted_html  # nothing highlighted
    assert report.unlocated_sentences == []


# --------------------------------------------------------------------------- #
# The money demo: example_hallucinated → 62%
# --------------------------------------------------------------------------- #


def test_worked_example_scores_62():
    source, answer = _load("example_hallucinated.json")
    report = check(source, answer, n=3, provider=MockProvider())
    assert report.faithfulness_score == 0.625
    assert report.n_supported == 5
    assert report.n_not_enough_info == 3
    assert report.n_contradicted == 0
    assert report.n_claims == 8
    assert report.n_low_confidence == 2  # claims 2 and 5 (split votes)
    assert report.n_refused == 0
    # Exactly three distinct source sentences render amber (claims 2, 5, 6).
    assert report.highlighted_html.count(f"background:{AMBER}") == 3
    assert report.unlocated_sentences == []


def test_worked_example_score_truncates_to_62_percent():
    # int(0.625 * 100) == 62 (the canonical headline; round() would also give 62 here
    # but is fragile — see CLI/Split-09 note).
    source, answer = _load("example_hallucinated.json")
    report = check(source, answer, n=3, provider=MockProvider())
    assert int(report.faithfulness_score * 100) == 62


# --------------------------------------------------------------------------- #
# The clean counter-example: example_grounded → 100%
# --------------------------------------------------------------------------- #


def test_grounded_example_scores_100():
    source, answer = _load("example_grounded.json")
    report = check(source, answer, n=3, provider=MockProvider())
    assert report.faithfulness_score == 1.0
    assert report.n_supported == report.n_claims == 10
    assert report.n_not_enough_info == 0
    assert report.n_contradicted == 0
    assert all(c.label == "SUPPORTED" for c in report.claims)
    assert report.unlocated_sentences == []


def test_no_verdict_key_is_source_substring():
    # The mock matches verdict keys against SOURCE+CLAIM text, so a key that is a
    # verbatim substring of the SOURCE would hijack every grounding call (it caused a
    # real regression in this split). Pin that no seeded verdict key appears in the
    # source — for either example (they share the source).
    from groundcheck import worked_example as we

    src = re.sub(r"\s+", " ", we.WORKED_EXAMPLE_SOURCE).strip().lower()
    keys = list(we.WORKED_EXAMPLE_VERDICTS) + list(we.GROUNDED_EXAMPLE_VERDICTS)
    offenders = [k for k in keys if re.sub(r"\s+", " ", k).strip().lower() in src]
    assert offenders == [], f"these verdict keys are source substrings: {offenders}"


def test_example_json_matches_pinned_fixtures():
    # The shipped JSON must equal the (dc-html-pinned) fixtures, or mock mode would
    # silently fall back to default verdicts and break the 62%.
    from groundcheck import worked_example as we

    hs, ha = _load("example_hallucinated.json")
    gs, ga = _load("example_grounded.json")
    assert hs == we.WORKED_EXAMPLE_SOURCE
    assert ha == we.WORKED_EXAMPLE_ANSWER
    assert gs == we.GROUNDED_EXAMPLE_SOURCE
    assert ga == we.GROUNDED_EXAMPLE_ANSWER


# --------------------------------------------------------------------------- #
# Document order under out-of-order completion
# --------------------------------------------------------------------------- #


class _JitterStub:
    """Decomposes into 5 indexed claims; grounding sleeps so earlier claims finish
    *last* — exercising out-of-order future completion. Thread-safe (no shared state)."""

    N_CLAIMS = 5

    def parse(self, *, output_model, user_blocks, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [
                DecomposedClaim(claim=f"claim_{i}", source_sentence=f"claim_{i}")
                for i in range(self.N_CLAIMS)
            ]
            return _ok(Decomposition(claims=claims))
        text = "".join(b.get("text", "") for b in user_blocks)
        idx = int(re.search(r"claim_(\d+)", text).group(1))
        time.sleep((self.N_CLAIMS - idx) * 0.003)  # idx 0 sleeps longest → resolves last
        return _ok(GroundingVerdict(label="SUPPORTED", supporting_span="", rationale=f"idx={idx}"))


def test_document_order_preserved():
    report = check("src", "answer with claims", n=1, provider=_JitterStub())
    assert [c.claim for c in report.claims] == [f"claim_{i}" for i in range(5)]


def test_progress_callback_fires_per_claim():
    seen: list[tuple[int, int]] = []
    check("src", "answer", n=1, provider=_JitterStub(), on_progress=lambda d, t: seen.append((d, t)))
    assert len(seen) == 5
    assert [d for d, _ in seen] == [1, 2, 3, 4, 5]  # monotonic done count
    assert all(t == 5 for _, t in seen)  # total is constant


# --------------------------------------------------------------------------- #
# Low-confidence + refusal surface; refused claim is NEI
# --------------------------------------------------------------------------- #


def test_low_confidence_and_refused_surface():
    answer = "Claim A about awareness levels. Claim B about another topic entirely."
    decomp = Decomposition(
        claims=[
            DecomposedClaim(
                claim="Claim A about awareness levels.",
                source_sentence="Claim A about awareness levels.",
            ),
            DecomposedClaim(
                claim="Claim B about another topic entirely.",
                source_sentence="Claim B about another topic entirely.",
            ),
        ]
    )
    provider = MockProvider(seed_worked_example=False)
    provider.register(answer, decomp)
    # Claim A: a split vote (2 NEI, 1 SUPPORTED) → low confidence.
    provider.register(
        "Claim A about awareness levels.",
        [
            GroundingVerdict(label="NOT_ENOUGH_INFO", supporting_span="", rationale="silent"),
            GroundingVerdict(label="NOT_ENOUGH_INFO", supporting_span="", rationale="silent"),
            GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="noise"),
        ],
    )
    # Claim B: the judge refuses every run → NEI + refused.
    provider.register("Claim B about another topic entirely.", REFUSAL)

    report = check("a source", answer, n=3, provider=provider)
    assert report.n_refused == 1
    assert report.n_low_confidence >= 1
    refused = [c for c in report.claims if c.refused]
    assert len(refused) == 1
    assert refused[0].label == "NOT_ENOUGH_INFO"


# --------------------------------------------------------------------------- #
# Cost is summed: decompose cost + per-run grounding costs
# --------------------------------------------------------------------------- #


class _CostStub:
    """Returns a fixed Sonnet usage for decompose (1 claim) and a fixed Opus usage
    for each grounding run, so the total cost is computable by hand."""

    def __init__(self, dec_usage: Usage, ground_usage: Usage) -> None:
        self.dec_usage = dec_usage
        self.ground_usage = ground_usage

    def parse(self, *, output_model, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [DecomposedClaim(claim="only claim", source_sentence="only claim")]
            return _ok(Decomposition(claims=claims), usage=self.dec_usage)
        return _ok(
            GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"),
            usage=self.ground_usage,
        )


def test_cost_is_summed():
    dec_usage = Usage(input_tokens=200, output_tokens=80)
    ground_usage = Usage(input_tokens=100, cache_read_input_tokens=20, output_tokens=30)
    n = 3
    report = check("a source", "an answer with one fact.", n=n, provider=_CostStub(dec_usage, ground_usage))

    expected_dec = compute_cost(DECOMPOSE_MODEL, dec_usage)
    expected_ground = n * compute_cost(GROUND_MODEL, ground_usage)
    assert report.cost_usd == pytest.approx(expected_dec + expected_ground)
    # The decompose cost is genuinely included (not dropped).
    assert report.cost_usd > expected_ground
    assert expected_dec > 0.0


# --------------------------------------------------------------------------- #
# A single claim's grounding failure degrades, never aborts the check
# --------------------------------------------------------------------------- #


class _ExplodingGroundStub:
    """Decomposes into 2 claims; one grounding call raises, the other succeeds."""

    def parse(self, *, output_model, user_blocks, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [
                DecomposedClaim(claim="good claim text", source_sentence="good claim text"),
                DecomposedClaim(claim="boom claim text", source_sentence="boom claim text"),
            ]
            return _ok(Decomposition(claims=claims))
        text = "".join(b.get("text", "") for b in user_blocks)
        if "boom" in text:
            raise RuntimeError("simulated grounding failure")
        return _ok(GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"))


def test_one_claim_failure_degrades_to_nei():
    report = check("src", "answer", n=2, provider=_ExplodingGroundStub())
    assert report.n_claims == 2  # both claims present, in order
    labels = {c.claim: c.label for c in report.claims}
    assert labels["good claim text"] == "SUPPORTED"
    assert labels["boom claim text"] == "NOT_ENOUGH_INFO"  # degraded, not a crash


# --------------------------------------------------------------------------- #
# Warm-one-first gate (spec §10)
# --------------------------------------------------------------------------- #


def test_warm_cache_gate_threshold():
    # Below the 4096-token floor → fan out immediately (no warm).
    small = "x" * 1000  # ~250 tokens
    assert _should_warm_cache_first(small) is False
    # At/above the floor → warm one first.
    big = "x" * (SOURCE_CACHE_FLOOR_TOKENS * 4 + 4)
    assert _should_warm_cache_first(big) is True


def test_demo_source_does_not_warm():
    source, _ = _load("example_hallucinated.json")
    assert _should_warm_cache_first(source) is False  # ~90 tokens → immediate fan-out


def test_large_source_still_correct_with_warm_path():
    # A source big enough to trigger the warm-one-first branch still produces a
    # correct, fully-grounded report (8 claims) — the branch only changes scheduling.
    from groundcheck import worked_example as we

    big_source = we.WORKED_EXAMPLE_SOURCE + (" Filler sentence to pad the source." * 1200)
    assert _should_warm_cache_first(big_source) is True
    report = check(big_source, we.WORKED_EXAMPLE_ANSWER, n=3, provider=MockProvider())
    assert report.faithfulness_score == 0.625
    assert report.n_claims == 8


# --------------------------------------------------------------------------- #
# Report carries everything the UI needs (UI spec §C2/§C3 contract)
# --------------------------------------------------------------------------- #


def test_report_carries_ui_fields():
    source, answer = _load("example_hallucinated.json")
    report = check(source, answer, n=3, provider=MockProvider())
    # Per-claim fields the mockup renders.
    for c in report.claims:
        assert isinstance(c.votes, dict) and c.votes
        assert 0.0 <= c.confidence <= 1.0
        assert isinstance(c.supporting_span, str)
        assert c.rationale
        assert isinstance(c.refused, bool)
    # Report-level fields.
    assert report.latency_s >= 0.0
    assert report.cost_usd > 0.0
    assert report.prompt_version == "v3"
    assert report.n_runs == 3
    assert report.highlighted_html  # non-empty highlighted answer


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _ok(parsed, *, usage: Usage | None = None) -> ParseResult:
    return ParseResult(
        parsed=parsed,
        usage=usage or Usage(input_tokens=5, output_tokens=5),
        stop_reason="end_turn",
        refused=False,
        truncated=False,
    )


# --------------------------------------------------------------------------- #
# Optional live smoke (needs a real key; skipped otherwise) — the demo headline
# --------------------------------------------------------------------------- #


@pytest.mark.api
def test_real_check_smoke(real_provider):
    source, answer = _load("example_hallucinated.json")
    report = check(source, answer, n=3, provider=real_provider)
    # Distributional, not exact: the answer is mostly-but-not-fully faithful, and the
    # real decompose splits more aggressively than the pinned mock (see PROGRESS §03).
    assert report.faithfulness_score is not None
    assert 0.4 <= report.faithfulness_score <= 0.8
    assert report.n_claims >= 6
    assert report.cost_usd > 0.0
