"""Tests for cost accounting + Usage mapping (Split 02). No key required."""

from __future__ import annotations

import pytest

from groundcheck.llm import Usage, compute_cost

# Anchor cases (hand-computed in SPLIT-02): three input buckets + output, with
# CACHE_WRITE_MULTIPLIER=1.25 and CACHE_READ_MULTIPLIER=0.1.
#   (model, input, cache_create, cache_read, output, expected_usd)
#
# NOTE on row 2: SPLIT-02's table prints the total as 0.029448, but its OWN listed
# components — 0.0005 + 0.0256 + 0.002048 + 0.00125 — sum to 0.029398. The split
# doc's hand-summed constant has a 5e-5 arithmetic slip; the formula (the real
# contract) and our code are correct, so we assert the true value 0.029398.
# Recorded as a divergence in PROGRESS.md.
ANCHORS = [
    ("claude-opus-4-8", 1000, 0, 0, 200, 0.010),
    ("claude-opus-4-8", 100, 4096, 4096, 50, 0.029398),
    ("claude-sonnet-4-6", 2000, 0, 0, 500, 0.0135),
    ("claude-sonnet-4-6", 0, 0, 8192, 0, 0.0024576),
]


@pytest.mark.parametrize("model, inp, cc, cr, out, expected", ANCHORS)
def test_cost_anchor_cases(model, inp, cc, cr, out, expected):
    usage = Usage(
        input_tokens=inp,
        cache_creation_input_tokens=cc,
        cache_read_input_tokens=cr,
        output_tokens=out,
    )
    assert compute_cost(model, usage) == pytest.approx(expected)


def test_cost_unknown_model_raises():
    with pytest.raises(KeyError):
        compute_cost("bogus", Usage())


def test_cost_zero_usage_is_zero():
    assert compute_cost("claude-opus-4-8", Usage()) == 0.0


def test_usage_from_anthropic_object():
    class _U:
        input_tokens = 100
        cache_creation_input_tokens = 4096
        cache_read_input_tokens = 4096
        output_tokens = 50

    usage = Usage.from_anthropic(_U())
    assert usage == Usage(
        input_tokens=100,
        cache_creation_input_tokens=4096,
        cache_read_input_tokens=4096,
        output_tokens=50,
    )


def test_usage_from_anthropic_missing_fields_default_to_zero():
    class _Partial:
        input_tokens = 42  # the cache + output fields are absent

    usage = Usage.from_anthropic(_Partial())
    assert usage == Usage(input_tokens=42)


def test_usage_from_anthropic_none_is_empty():
    assert Usage.from_anthropic(None) == Usage()


def test_usage_from_openai_splits_cached_from_fresh():
    class _Details:
        cached_tokens = 4096

    class _U:
        prompt_tokens = 5000  # 4096 cached + 904 fresh
        completion_tokens = 120
        prompt_tokens_details = _Details()

    usage = Usage.from_openai(_U())
    assert usage == Usage(
        input_tokens=904,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=4096,
        output_tokens=120,
    )


def test_usage_from_openai_no_cache_details():
    usage = Usage.from_openai({"prompt_tokens": 300, "completion_tokens": 80})
    assert usage == Usage(input_tokens=300, cache_read_input_tokens=0, output_tokens=80)
