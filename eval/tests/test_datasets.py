"""Tests for the gold datasets + ``load_dataset`` (Split 07). No key needed.

These pin the dataset shape (counts / balance / topics / frozen slice / buckets),
the byte-identity of the two Split-05 example cases, the no-empty-answer rule, and
the loader's malformed-item contract. Every expectation is the spec's, never the
implementation's.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.run import (
    DATASETS_DIR,
    TIER2_BUCKETS,
    Tier1Item,
    Tier2Item,
    load_dataset,
)

LABELS = {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
_CORE_EXAMPLES = Path(__file__).resolve().parents[2] / "core" / "examples"


# --------------------------------------------------------------------------- #
# Tier 1
# --------------------------------------------------------------------------- #


def test_tier1_counts_and_balance():
    """~45 items, ~15 per label (band 12-18), valid labels, unique ids, >=4 topics."""
    ds = load_dataset(DATASETS_DIR / "tier1_claims.yaml")
    items = ds.items
    assert not ds.skipped, f"unexpected skipped items: {ds.skipped}"
    assert 42 <= len(items) <= 48, f"expected ~45 triples, got {len(items)}"

    labels = Counter(it.gold_label for it in items)
    assert set(labels) == LABELS
    for label in LABELS:
        assert 12 <= labels[label] <= 18, f"{label} count {labels[label]} outside 12-18"

    ids = [it.id for it in items]
    assert len(set(ids)) == len(ids), "Tier-1 ids must be unique"

    topics = {it.topic for it in items}
    assert len(topics) >= 4, f"need >=4 distinct topics (Appendix B.1), got {topics}"


def test_tier1_frozen_slice():
    """~20% (band 15-25%) marked frozen, stratified (>=2 per class)."""
    ds = load_dataset(DATASETS_DIR / "tier1_claims.yaml")
    frozen = [it for it in ds.items if it.frozen]
    frac = len(frozen) / len(ds.items)
    assert 0.15 <= frac <= 0.25, f"frozen slice {frac:.0%} outside 15-25%"

    by_label = Counter(it.gold_label for it in frozen)
    for label in LABELS:
        assert by_label[label] >= 2, f"frozen slice not stratified: {label}={by_label[label]}"


def test_tier1_includes_appendix_b1_worked_examples():
    """The six Appendix B.1 worked (claim, gold_label) pairs are present verbatim."""
    ds = load_dataset(DATASETS_DIR / "tier1_claims.yaml")
    present = {(it.claim, it.gold_label) for it in ds.items}
    expected = {
        ("A reading of 130/80 mm Hg or higher is high blood pressure.", "SUPPORTED"),
        ("Cutting salt lowers blood pressure by exactly 25% in every patient.", "NOT_ENOUGH_INFO"),
        ("High blood pressure has no effect on stroke risk.", "CONTRADICTED"),
        ("Everyone with hypertension needs medication.", "CONTRADICTED"),
        ("Hypertension is typically symptomless.", "SUPPORTED"),
        ("Hypertension is the leading cause of death worldwide.", "NOT_ENOUGH_INFO"),
    }
    missing = expected - present
    assert not missing, f"missing Appendix B.1 worked examples: {missing}"


# --------------------------------------------------------------------------- #
# Tier 2
# --------------------------------------------------------------------------- #


def test_tier2_buckets():
    """~18 items matching the B.2 bucket counts; valid gold/bucket; unique ids; ~20% frozen."""
    ds = load_dataset(DATASETS_DIR / "tier2_answers.yaml")
    items = ds.items
    assert not ds.skipped, f"unexpected skipped items: {ds.skipped}"
    assert 16 <= len(items) <= 20, f"expected ~18 answers, got {len(items)}"

    buckets = Counter(it.bucket for it in items)
    assert set(buckets) <= set(TIER2_BUCKETS)
    # B.2 target counts (within a +/-1 tolerance band).
    expected = {
        "fully_grounded": 5,
        "single_fabricated_stat": 4,
        "direct_contradiction": 4,
        "faithful_paraphrase": 3,
        "mixed": 2,
    }
    for bucket, target in expected.items():
        assert abs(buckets[bucket] - target) <= 1, f"{bucket}={buckets[bucket]} far from ~{target}"

    for it in items:
        assert isinstance(it.gold_is_faithful, bool)

    ids = [it.id for it in items]
    assert len(set(ids)) == len(ids), "Tier-2 ids must be unique"

    frozen = [it for it in items if it.frozen]
    frac = len(frozen) / len(items)
    assert 0.15 <= frac <= 0.30, f"frozen slice {frac:.0%} outside ~20%"


def test_tier2_bucket_gold_consistency():
    """Faithful buckets are gold=true; unfaithful buckets are gold=false (Appendix B.2)."""
    ds = load_dataset(DATASETS_DIR / "tier2_answers.yaml")
    faithful_buckets = {"fully_grounded", "faithful_paraphrase"}
    unfaithful_buckets = {"single_fabricated_stat", "direct_contradiction", "mixed"}
    for it in ds.items:
        if it.bucket in faithful_buckets:
            assert it.gold_is_faithful is True, f"{it.id} ({it.bucket}) should be faithful"
        elif it.bucket in unfaithful_buckets:
            assert it.gold_is_faithful is False, f"{it.id} ({it.bucket}) should be unfaithful"


def test_tier2_examples_are_first_two():
    """t2-001/t2-002 source+answer equal the Split-05 example JSONs, byte-for-byte."""
    ds = load_dataset(DATASETS_DIR / "tier2_answers.yaml")
    by_id = {it.id: it for it in ds.items}

    hall = json.loads((_CORE_EXAMPLES / "example_hallucinated.json").read_text(encoding="utf-8"))
    grnd = json.loads((_CORE_EXAMPLES / "example_grounded.json").read_text(encoding="utf-8"))

    assert by_id["t2-001"].source == hall["source"]
    assert by_id["t2-001"].answer == hall["answer"]
    assert by_id["t2-001"].gold_is_faithful is False
    assert by_id["t2-002"].source == grnd["source"]
    assert by_id["t2-002"].answer == grnd["answer"]
    assert by_id["t2-002"].gold_is_faithful is True


def test_tier2_every_case_has_checkable_claim():
    """No Tier-2 answer is empty/whitespace (the 0-claim exclusion, §11 corner case (b))."""
    ds = load_dataset(DATASETS_DIR / "tier2_answers.yaml")
    for it in ds.items:
        assert it.answer.strip(), f"{it.id} has an empty/whitespace answer"


# --------------------------------------------------------------------------- #
# load_dataset contract
# --------------------------------------------------------------------------- #


def test_load_dataset_rejects_malformed(tmp_path):
    """Bad label / missing field items are flagged into `skipped`, not crashing the load."""
    bad = tmp_path / "bad_tier1.yaml"
    bad.write_text(
        "\n".join(
            [
                "- id: ok-1",
                "  topic: t",
                "  source: s",
                "  claim: c",
                "  gold_label: SUPPORTED",
                "- id: bad-label",
                "  topic: t",
                "  source: s",
                "  claim: c",
                "  gold_label: MAYBE",  # invalid label
                "- id: missing-claim",
                "  topic: t",
                "  source: s",
                "  gold_label: SUPPORTED",  # missing required 'claim'
            ]
        ),
        encoding="utf-8",
    )
    ds = load_dataset(bad)
    assert len(ds.items) == 1 and ds.items[0].id == "ok-1"
    skipped_ids = {raw.get("id") for raw, _reason in ds.skipped}
    assert skipped_ids == {"bad-label", "missing-claim"}


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    """A duplicate id is a structural error (corrupts gold↔pred alignment) → ValueError."""
    dup = tmp_path / "dup.yaml"
    dup.write_text(
        "\n".join(
            [
                "- id: t1-x",
                "  topic: t",
                "  source: s",
                "  claim: c",
                "  gold_label: SUPPORTED",
                "- id: t1-x",
                "  topic: t",
                "  source: s",
                "  claim: c2",
                "  gold_label: CONTRADICTED",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset(dup)


def test_load_dataset_rejects_non_list(tmp_path):
    """A non-list / empty file is rejected outright."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(empty)
