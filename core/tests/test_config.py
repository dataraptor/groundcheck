"""Tests for the config constants (Split 01)."""

from __future__ import annotations

from groundcheck import config


def test_model_ids():
    assert config.DECOMPOSE_MODEL == "claude-sonnet-4-6"
    assert config.GROUND_MODEL == "claude-opus-4-8"


def test_pricing_values():
    # per-token == per-MTok / 1e6 (Opus $5/$25, Sonnet $3/$15)
    assert config.PRICING[config.GROUND_MODEL]["input"] == 5 / 1e6
    assert config.PRICING[config.GROUND_MODEL]["output"] == 25 / 1e6
    assert config.PRICING[config.DECOMPOSE_MODEL]["input"] == 3 / 1e6
    assert config.PRICING[config.DECOMPOSE_MODEL]["output"] == 15 / 1e6


def test_severity_order():
    assert config.SEVERITY_ORDER["CONTRADICTED"] > config.SEVERITY_ORDER["NOT_ENOUGH_INFO"]
    assert config.SEVERITY_ORDER["NOT_ENOUGH_INFO"] > config.SEVERITY_ORDER["SUPPORTED"]
