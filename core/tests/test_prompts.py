"""Tests for the canonical prompts (Split 01)."""

from __future__ import annotations

from groundcheck.prompts import (
    DECOMPOSE_SYSTEM,
    GROUND_SYSTEM,
    PROMPT_VERSION,
    decompose_user,
    ground_claim_block,
    ground_source_block,
)


def test_prompt_version():
    assert PROMPT_VERSION == "v3"


def test_decompose_prompt_pins():
    assert "atomic factual claims" in DECOMPOSE_SYSTEM
    assert "IGNORE anything that is not a checkable factual assertion" in DECOMPOSE_SYSTEM
    assert "return an empty list" in DECOMPOSE_SYSTEM


def test_ground_prompt_pins():
    assert "strict grounding verifier" in GROUND_SYSTEM
    assert "Judge ONLY against the SOURCE" in GROUND_SYSTEM
    for label in ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"):
        assert label in GROUND_SYSTEM


def test_user_builders():
    assert decompose_user("X") == "ANSWER:\nX"
    assert ground_source_block("S") == "SOURCE:\nS"
    assert ground_claim_block("C") == "\n\nCLAIM:\nC"
