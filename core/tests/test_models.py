"""Tests for the data contracts (Split 01)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from groundcheck.models import (
    ClaimResult,
    DecomposedClaim,
    Decomposition,
    FaithfulnessReport,
    GroundingVerdict,
)

# The JSON-schema keywords that Anthropic structured outputs reject (README API facts).
BANNED_SCHEMA_KEYWORDS = ("minLength", "maxLength", "minimum", "maximum", "multipleOf")


@pytest.mark.parametrize("model", [Decomposition, GroundingVerdict, DecomposedClaim])
def test_sent_schemas_have_no_unsupported_keywords(model):
    schema_str = json.dumps(model.model_json_schema())
    for kw in BANNED_SCHEMA_KEYWORDS:
        assert kw not in schema_str, f"{model.__name__} schema contains banned keyword {kw!r}"


def test_grounding_verdict_rejects_bad_label():
    with pytest.raises(ValidationError):
        GroundingVerdict(label="MAYBE", supporting_span="", rationale="nope")


def test_models_roundtrip():
    samples = [
        DecomposedClaim(claim="BP can be high", source_sentence="BP can be high."),
        Decomposition(
            claims=[DecomposedClaim(claim="c1", source_sentence="s1.")]
        ),
        GroundingVerdict(
            label="SUPPORTED",
            supporting_span="usually has no symptoms",
            rationale="The source says it usually has no symptoms.",
        ),
        ClaimResult(
            claim="c1",
            source_sentence="s1.",
            label="NOT_ENOUGH_INFO",
            supporting_span="",
            rationale="Source is silent.",
            votes={"NOT_ENOUGH_INFO": 2, "SUPPORTED": 1},
            confidence=0.67,
            refused=False,
        ),
        FaithfulnessReport.from_claims(
            [
                ClaimResult(
                    claim="c1",
                    source_sentence="s1.",
                    label="SUPPORTED",
                    supporting_span="span",
                    rationale="ok",
                    votes={"SUPPORTED": 3},
                    confidence=1.0,
                )
            ],
            cost_usd=0.0012,
            latency_s=1.5,
            n_runs=3,
            prompt_version="v3",
        ),
    ]
    for obj in samples:
        restored = type(obj).model_validate_json(obj.model_dump_json())
        assert restored == obj


def test_report_from_claims_empty():
    report = FaithfulnessReport.from_claims(
        [], cost_usd=0.0, latency_s=0.0, n_runs=3, prompt_version="v3"
    )
    assert report.faithfulness_score is None
    assert report.n_claims == 0
    assert report.n_supported == 0
    assert report.n_contradicted == 0
    assert report.n_not_enough_info == 0
    assert report.n_low_confidence == 0
    assert report.n_refused == 0


def _claim(label, confidence=1.0, refused=False):
    return ClaimResult(
        claim="c",
        source_sentence="s.",
        label=label,
        supporting_span="" if label == "NOT_ENOUGH_INFO" else "span",
        rationale="r",
        votes={label: 3},
        confidence=confidence,
        refused=refused,
    )


def test_report_from_claims_counts_and_score():
    claims = (
        [_claim("SUPPORTED") for _ in range(5)]
        # two of the NEI claims are borderline / refusal-affected
        + [_claim("NOT_ENOUGH_INFO", confidence=0.67)]
        + [_claim("NOT_ENOUGH_INFO", confidence=0.67, refused=True)]
        + [_claim("NOT_ENOUGH_INFO")]
    )
    report = FaithfulnessReport.from_claims(
        claims, cost_usd=0.01, latency_s=2.0, n_runs=3, prompt_version="v3"
    )
    assert report.n_claims == 8
    assert report.n_supported == 5
    assert report.n_not_enough_info == 3
    assert report.n_contradicted == 0
    assert report.faithfulness_score == 0.625  # 5/8
    assert report.n_low_confidence == 2  # the two confidence-0.67 claims
    assert report.n_refused == 1  # the one refused=True claim
