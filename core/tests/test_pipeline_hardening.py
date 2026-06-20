"""Split 11 — pipeline hardening regressions (spec §17, no key).

These extend the Split-05 ``test_pipeline.py`` checks with the *combined* stress the
Split-11 brief calls out, so a future engine change cannot silently regress them:

* **document order + per-claim failure together** — claims resolve out of order AND
  one claim's grounding raises; the report must stay in document order and the failed
  claim must degrade to NOT_ENOUGH_INFO *with a warning*, never aborting the check;
* **concurrency sanity at scale** — many claims through the default pool still come
  back deterministically in document order;
* **the oversize chain** — an over-cap source (and answer) warns + truncates rather
  than ballooning the Opus ×N×claims cost or overflowing context (the source cap,
  ``MAX_SOURCE_TOKENS``, was defined but un-enforced before this split — see PROGRESS
  Split 11 bug provenance; the answer cap already lived in ``decompose``).

Every test injects a stub/mock provider, so the suite never touches the network.
"""

from __future__ import annotations

import logging
import re
import time

import pytest

from groundcheck import check
from groundcheck.config import MAX_ANSWER_TOKENS, MAX_SOURCE_TOKENS
from groundcheck.llm import ParseResult, Usage
from groundcheck.models import DecomposedClaim, Decomposition, GroundingVerdict
from groundcheck.pipeline import _truncate_source_if_oversized

_CHARS_PER_TOKEN = 4  # mirrors pipeline/decompose's cheap heuristic


def _ok(parsed, *, usage: Usage | None = None) -> ParseResult:
    return ParseResult(
        parsed=parsed,
        usage=usage or Usage(input_tokens=5, output_tokens=5),
        stop_reason="end_turn",
        refused=False,
        truncated=False,
    )


# --------------------------------------------------------------------------- #
# 1. Document order + a single per-claim failure, TOGETHER (the §5/§10 stress)
# --------------------------------------------------------------------------- #


class _JitterAndFailStub:
    """Decomposes into N indexed claims; grounding jitters (earlier claims finish
    *last*) AND one chosen claim's grounding raises — so the report must both
    preserve document order and degrade the failed claim, in one run.
    """

    def __init__(self, n_claims: int = 12, fail_index: int = 7) -> None:
        self.n_claims = n_claims
        self.fail_index = fail_index

    def parse(self, *, output_model, user_blocks=None, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [
                DecomposedClaim(claim=f"claim_{i}", source_sentence=f"claim_{i}")
                for i in range(self.n_claims)
            ]
            return _ok(Decomposition(claims=claims))
        text = "".join(b.get("text", "") for b in (user_blocks or []))
        idx = int(re.search(r"claim_(\d+)", text).group(1))
        # Earlier indices sleep longer → they resolve last (out-of-order completion).
        time.sleep((self.n_claims - idx) * 0.002)
        if idx == self.fail_index:
            raise RuntimeError(f"simulated grounding failure for claim_{idx}")
        return _ok(GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale=f"idx={idx}"))


def test_out_of_order_with_one_failure_stays_ordered_and_degrades(caplog):
    stub = _JitterAndFailStub(n_claims=12, fail_index=7)
    with caplog.at_level(logging.WARNING, logger="groundcheck"):
        report = check("src", "answer with many claims", n=2, provider=stub)

    # Document order is preserved despite out-of-order completion.
    assert [c.claim for c in report.claims] == [f"claim_{i}" for i in range(12)]
    # Exactly the failed claim degraded to NEI; everything else is SUPPORTED.
    by_claim = {c.claim: c.label for c in report.claims}
    assert by_claim["claim_7"] == "NOT_ENOUGH_INFO"
    assert all(by_claim[f"claim_{i}"] == "SUPPORTED" for i in range(12) if i != 7)
    assert report.n_claims == 12
    assert report.n_supported == 11
    assert report.n_not_enough_info == 1
    # The failure was surfaced as a warning, not swallowed and not raised.
    assert any(
        "grounding failed" in r.getMessage() and "claim_7" in r.getMessage()
        for r in caplog.records
    ), "the degraded claim must emit a warning naming the claim"


def test_per_claim_failure_does_not_abort_the_check():
    # Even with the failure at index 0 (the warm-path candidate would be claim 0 on a
    # big source; here the source is small so it fans out), the check still completes.
    report = check("src", "answer", n=3, provider=_JitterAndFailStub(n_claims=4, fail_index=0))
    assert report.n_claims == 4
    assert report.faithfulness_score is not None  # produced a real score, no crash


# --------------------------------------------------------------------------- #
# 2. Concurrency sanity at scale — many claims, default pool, ordered result
# --------------------------------------------------------------------------- #


class _ScaleJitterStub:
    """Decomposes into ``n_claims`` claims; each grounding sleeps a pseudo-random
    (index-derived, deterministic) amount so completion order is scrambled."""

    def __init__(self, n_claims: int = 30) -> None:
        self.n_claims = n_claims

    def parse(self, *, output_model, user_blocks=None, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [
                DecomposedClaim(claim=f"c{i:03d}", source_sentence=f"c{i:03d}")
                for i in range(self.n_claims)
            ]
            return _ok(Decomposition(claims=claims))
        text = "".join(b.get("text", "") for b in (user_blocks or []))
        idx = int(re.search(r"c(\d+)", text).group(1))
        # A scrambled-but-deterministic delay (no Math.random / wall clock dependence).
        time.sleep(((idx * 7) % 11) * 0.001)
        return _ok(GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale=str(idx)))


def test_many_claims_complete_in_document_order():
    report = check("src", "many", n=2, provider=_ScaleJitterStub(n_claims=30))
    assert [c.claim for c in report.claims] == [f"c{i:03d}" for i in range(30)]
    assert report.n_supported == 30
    assert report.faithfulness_score == 1.0


# --------------------------------------------------------------------------- #
# 3. The oversize chain — source AND answer caps enforced + warned (spec §17)
# --------------------------------------------------------------------------- #


def test_truncate_source_unit():
    # Unit-level: under the cap is returned unchanged; over the cap is cut to the cap.
    max_chars = MAX_SOURCE_TOKENS * _CHARS_PER_TOKEN
    small = "x" * 100
    assert _truncate_source_if_oversized(small) is small
    big = "y" * (max_chars + 5000)
    cut = _truncate_source_if_oversized(big)
    assert len(cut) == max_chars
    assert big.startswith(cut)


class _SourceRecordingStub:
    """One claim; records the SOURCE length seen by each grounding call so we can
    prove the pipeline truncated the source *before* spending grounding calls."""

    def __init__(self) -> None:
        self.source_lens: list[int] = []

    def parse(self, *, output_model, user_blocks=None, **kwargs) -> ParseResult:
        if output_model is Decomposition:
            claims = [DecomposedClaim(claim="only claim", source_sentence="only claim")]
            return _ok(Decomposition(claims=claims))
        # The SOURCE block is "SOURCE:\n<source>"; recover the source length.
        block = next(b.get("text", "") for b in (user_blocks or []) if b.get("text", "").startswith("SOURCE:"))
        self.source_lens.append(len(block) - len("SOURCE:\n"))
        return _ok(GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok"))


def test_oversize_source_truncates_and_warns(caplog):
    max_chars = MAX_SOURCE_TOKENS * _CHARS_PER_TOKEN
    oversize_source = "Long source sentence. " * 4000  # well over 64000 chars
    assert len(oversize_source) > max_chars
    stub = _SourceRecordingStub()

    with caplog.at_level(logging.WARNING, logger="groundcheck"):
        report = check(oversize_source, "An answer with one fact.", n=2, provider=stub)

    # The warning was surfaced (never silent — spec §17).
    msgs = [r.getMessage() for r in caplog.records]
    assert any("source is ~" in m and "cap" in m for m in msgs), msgs
    # And the source that actually reached grounding was capped, not the full balloon.
    assert stub.source_lens, "grounding should have been called"
    assert all(length <= max_chars for length in stub.source_lens)
    # The check still produced a valid report (no crash).
    assert report.faithfulness_score is not None
    assert report.n_claims == 1


def test_oversize_answer_truncates_and_warns(caplog):
    max_chars = MAX_ANSWER_TOKENS * _CHARS_PER_TOKEN
    oversize_answer = "This is a filler sentence. " * 4000
    assert len(oversize_answer) > max_chars
    stub = _SourceRecordingStub()

    with caplog.at_level(logging.WARNING, logger="groundcheck"):
        report = check("A short source.", oversize_answer, n=1, provider=stub)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("answer is ~" in m and "cap" in m for m in msgs), msgs
    # Still a valid report; the over-cap answer did not crash the check.
    assert report is not None
    assert "faithfulness_score" in report.model_dump()


def test_within_cap_inputs_emit_no_truncation_warning(caplog):
    stub = _SourceRecordingStub()
    with caplog.at_level(logging.WARNING, logger="groundcheck"):
        check("A short source.", "A short answer.", n=1, provider=stub)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("over the" in m and "cap" in m for m in msgs), (
        f"normal-size inputs must not warn about truncation: {msgs}"
    )
