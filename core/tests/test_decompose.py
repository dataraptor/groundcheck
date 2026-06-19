"""Tests for Split 03 — the decompose step (no key; one optional @api smoke).

Every no-key test injects a provider (a spy/stub or :class:`MockProvider`) so the
suite never touches the network and never depends on what env vars are set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from groundcheck.config import DECOMPOSE_MODEL, MAX_ANSWER_TOKENS
from groundcheck.decompose import DecomposeOutcome, _CHARS_PER_TOKEN, decompose
from groundcheck.llm import MockProvider, ParseResult, Usage, compute_cost
from groundcheck.models import DecomposedClaim, Decomposition
from groundcheck.worked_example import (
    WORKED_EXAMPLE_ANSWER,
    WORKED_EXAMPLE_CLAIMS,
    WORKED_EXAMPLE_SOURCE,
)

EXPECTED_CLAIMS = [
    "Hypertension usually causes no symptoms.",
    "Many people don't know they have it.",
    "A reading of 130/80 mm Hg or higher is considered high blood pressure.",
    "It increases the risk of heart attack, stroke, and kidney disease.",
    "Cutting salt lowers blood pressure by exactly 25% in every patient.",
    "Hypertension is the leading cause of death worldwide.",
    "Doctors recommend regular exercise and maintaining a healthy weight.",
    "Some patients need medication.",
]


# --------------------------------------------------------------------------- #
# Test doubles (dependency injection — no network)
# --------------------------------------------------------------------------- #


class _SpyProvider:
    """Records every parse() call; returns a canned ParseResult (default: empty)."""

    def __init__(self, result: ParseResult | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result

    def parse(self, **kwargs) -> ParseResult:
        self.calls.append(kwargs)
        if self._result is not None:
            return self._result
        return ParseResult(
            parsed=Decomposition(claims=[]),
            usage=Usage(),
            stop_reason="end_turn",
            refused=False,
            truncated=False,
        )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Empty / whitespace short-circuit (no spend)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("answer", ["", "   \n\t  ", "\n"])
def test_empty_answer_returns_no_claims_without_calling_provider(answer):
    spy = _SpyProvider()
    out = decompose(answer, provider=spy)
    assert isinstance(out, DecomposeOutcome)
    assert out.decomposition.claims == []
    assert out.cost_usd == 0.0
    assert spy.calls == []  # provider must NOT be called on the empty path


# --------------------------------------------------------------------------- #
# Worked example (mock) → the 8 §5 claims
# --------------------------------------------------------------------------- #


def test_worked_example_decomposes_to_8_claims():
    out = decompose(WORKED_EXAMPLE_ANSWER, provider=MockProvider())
    assert len(out.decomposition.claims) == 8
    assert [c.claim for c in out.decomposition.claims] == EXPECTED_CLAIMS


def test_source_sentences_present():
    out = decompose(WORKED_EXAMPLE_ANSWER, provider=MockProvider())
    answer_norm = _norm(WORKED_EXAMPLE_ANSWER)
    for claim in out.decomposition.claims:
        assert claim.source_sentence.strip()  # non-empty
        assert _norm(claim.source_sentence) in answer_norm  # whitespace-insensitive substring


def test_decompose_is_deterministic():
    a = decompose(WORKED_EXAMPLE_ANSWER, provider=MockProvider())
    b = decompose(WORKED_EXAMPLE_ANSWER, provider=MockProvider())
    assert a.decomposition.model_dump() == b.decomposition.model_dump()


def test_decompose_output_carries_no_verdict_fields():
    # Scope guard: decompose produces only {claim, source_sentence}; labels/verdicts
    # are Split 04. The contract model must not leak later-scope fields.
    assert set(DecomposedClaim.model_fields) == {"claim", "source_sentence"}


# --------------------------------------------------------------------------- #
# Cost is carried for the pipeline (Split 05)
# --------------------------------------------------------------------------- #


def test_cost_is_carried():
    usage = Usage(
        input_tokens=123,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=20,
        output_tokens=45,
    )
    result = ParseResult(
        parsed=Decomposition(claims=[DecomposedClaim(claim="x", source_sentence="x.")]),
        usage=usage,
        stop_reason="end_turn",
        refused=False,
        truncated=False,
    )
    out = decompose("some answer with a fact.", provider=_SpyProvider(result))
    assert out.cost_usd == compute_cost(DECOMPOSE_MODEL, usage)
    assert out.cost_usd > 0.0


# --------------------------------------------------------------------------- #
# Truncation / refusal degrade gracefully (warn, never crash)
# --------------------------------------------------------------------------- #


def test_truncation_warns_not_crashes(caplog):
    partial = Decomposition(claims=[DecomposedClaim(claim="partial", source_sentence="partial.")])
    result = ParseResult(
        parsed=partial,
        usage=Usage(input_tokens=5, output_tokens=5),
        stop_reason="max_tokens",
        refused=False,
        truncated=True,
    )
    with caplog.at_level(logging.WARNING):
        out = decompose("a long answer", provider=_SpyProvider(result))
    assert out.truncated is True
    assert out.decomposition.claims == partial.claims  # best-effort partial returned
    assert out.cost_usd > 0.0
    assert "max_tokens" in caplog.text


def test_truncation_with_no_parsed_returns_empty(caplog):
    result = ParseResult(
        parsed=None,
        usage=Usage(input_tokens=5, output_tokens=0),
        stop_reason="max_tokens",
        refused=False,
        truncated=True,
    )
    with caplog.at_level(logging.WARNING):
        out = decompose("a long answer", provider=_SpyProvider(result))
    assert out.truncated is True
    assert out.decomposition.claims == []  # degrades to empty, no crash


def test_refusal_warns_returns_empty(caplog):
    result = ParseResult(
        parsed=None,
        usage=Usage(input_tokens=5, output_tokens=0),
        stop_reason="refusal",
        refused=True,
        truncated=False,
    )
    with caplog.at_level(logging.WARNING):
        out = decompose("a benign answer", provider=_SpyProvider(result))
    assert out.refused is True
    assert out.decomposition.claims == []
    assert out.cost_usd > 0.0  # the refused call still cost money
    assert "refus" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# Oversized answer → truncate to the cap with a warning (spec §17)
# --------------------------------------------------------------------------- #


def test_oversized_answer_truncates_with_warning(caplog):
    max_chars = MAX_ANSWER_TOKENS * _CHARS_PER_TOKEN
    big = "fact. " * ((max_chars // 6) + 100)  # comfortably over the cap
    assert len(big) > max_chars
    spy = _SpyProvider()
    with caplog.at_level(logging.WARNING):
        out = decompose(big, provider=spy)
    assert isinstance(out, DecomposeOutcome)  # never crashes
    assert len(spy.calls) == 1
    sent_text = spy.calls[0]["user_blocks"][0]["text"]
    answer_sent = sent_text[len("ANSWER:\n") :]
    assert len(answer_sent) == max_chars  # provider received the truncated answer
    assert "truncat" in caplog.text.lower() and "cap" in caplog.text.lower()


def test_at_cap_answer_is_not_truncated():
    max_chars = MAX_ANSWER_TOKENS * _CHARS_PER_TOKEN
    exact = "a" * max_chars
    spy = _SpyProvider()
    decompose(exact, provider=spy)
    sent_text = spy.calls[0]["user_blocks"][0]["text"]
    assert sent_text[len("ANSWER:\n") :] == exact  # exactly at the cap → untouched


# --------------------------------------------------------------------------- #
# Worked-example fixture stays byte-identical to the frontend (anti-drift)
# --------------------------------------------------------------------------- #


def test_fixture_matches_dc_html_literals():
    dc_html = Path(__file__).resolve().parents[2] / "app" / "GroundCheck.dc.html"
    if not dc_html.exists():
        pytest.skip("app/GroundCheck.dc.html not present in this checkout")
    text = dc_html.read_text(encoding="utf-8")
    src = re.search(r'this\.SRC\s*=\s*"((?:[^"\\]|\\.)*)"', text)
    ans = re.search(r'this\.ANS\s*=\s*"((?:[^"\\]|\\.)*)"', text)
    assert src and ans, "could not find this.SRC / this.ANS in the mockup"
    assert src.group(1) == WORKED_EXAMPLE_SOURCE
    assert ans.group(1) == WORKED_EXAMPLE_ANSWER


def test_fixture_claims_match_expected():
    assert [c.claim for c in WORKED_EXAMPLE_CLAIMS] == EXPECTED_CLAIMS
    # In this worked example each claim's source_sentence == its answer sentence.
    for c in WORKED_EXAMPLE_CLAIMS:
        assert c.source_sentence == c.claim
        assert _norm(c.source_sentence) in _norm(WORKED_EXAMPLE_ANSWER)


# --------------------------------------------------------------------------- #
# Optional live smoke (needs a real key; skipped otherwise)
# --------------------------------------------------------------------------- #


@pytest.mark.api
def test_real_decompose_smoke(real_provider):
    out = decompose(WORKED_EXAMPLE_ANSWER, provider=real_provider)
    # Decomposition varies run-to-run — assert a loose band, not exact texts.
    assert len(out.decomposition.claims) >= 6
    assert out.cost_usd > 0.0
    for c in out.decomposition.claims:
        assert c.claim.strip()
        assert c.source_sentence.strip()
