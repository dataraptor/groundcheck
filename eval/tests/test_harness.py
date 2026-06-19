"""Tests for the harness logic (Split 07) — all via the ScriptedProvider, no key.

These drive the harness with a deterministic provider so the confusion matrix (and
thus every metric) is known in advance, then assert the harness reports exactly the
Split-06 hand-worked numbers. They also pin the τ sweep, both §11 corner cases, the
held-in/frozen split, mean ± spread aggregation, JSONL persistence, and refusal
surfacing. The ``@api`` smoke at the bottom runs the real provider and is skipped
without a key.
"""

from __future__ import annotations

import json

import pytest

from eval.run import (
    LoadedDataset,
    Tier1Item,
    Tier2Item,
    predicted_unfaithful,
    run,
    run_tier1,
    run_tier2,
)
from eval.tests.conftest import REFUSAL, ScriptedProvider
from groundcheck.models import (
    ClaimResult,
    DecomposedClaim,
    Decomposition,
    FaithfulnessReport,
)
from groundcheck.prompts import PROMPT_VERSION

S, C, N = "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _t1(item_id, claim, gold, *, frozen=False, marker=None):
    return Tier1Item(
        id=item_id,
        topic="scripted",
        source="Scripted source paragraph.",
        claim=f"Scripted claim {claim}.",
        gold_label=gold,
        frozen=frozen,
    )


def _t2_case(item_id, prefix, n_supported, n_total, *, gold_is_faithful, bucket, frozen=False):
    """A Tier-2 case that decomposes to ``n_total`` claims, ``n_supported`` SUPPORTED.

    Returns ``(item, decompose_entries, ground_entries)`` to feed a ScriptedProvider.
    The resulting ``check`` score is ``n_supported / n_total`` and ``n_contradicted == 0``.
    """
    answer = f"Scripted answer {prefix} body text."
    claims = [
        DecomposedClaim(claim=f"Scripted {prefix} claim {j}.", source_sentence=f"Scripted {prefix} claim {j}.")
        for j in range(1, n_total + 1)
    ]
    ground = {
        f"{prefix} claim {j}": (S if j <= n_supported else N) for j in range(1, n_total + 1)
    }
    item = Tier2Item(
        id=item_id,
        topic="scripted",
        source="Scripted source paragraph.",
        answer=answer,
        gold_is_faithful=gold_is_faithful,
        bucket=bucket,
        frozen=frozen,
    )
    return item, {answer: Decomposition(claims=claims)}, ground


def _claim(label):
    return ClaimResult(
        claim="c",
        source_sentence="s",
        label=label,
        supporting_span="",
        rationale="r",
        votes={label: 1},
        confidence=1.0,
    )


def _report(claims):
    return FaithfulnessReport.from_claims(
        claims, cost_usd=0.0, latency_s=0.0, n_runs=1, prompt_version=PROMPT_VERSION
    )


# --------------------------------------------------------------------------- #
# Tier 1 — metrics match a scripted confusion matrix (the Split-06 fixture)
# --------------------------------------------------------------------------- #


def test_tier1_metrics_match_scripted_confusion():
    """Scripted to reproduce the §06 10-item fixture → exact macro-F1 / accuracy / κ."""
    gold = [S, S, S, C, C, C, N, N, N, N]
    pred = [S, S, C, C, C, N, N, N, N, S]

    items = []
    ground_script = {}
    for i, (g, p) in enumerate(zip(gold, pred)):
        marker = f"m{i:02d}"  # m00..m09 — no marker is a substring of another
        items.append(_t1(f"t1-{i:02d}", marker, g))
        ground_script[marker] = p

    provider = ScriptedProvider(ground=ground_script)
    result = run_tier1(items, n=1, repeats=1, provider=provider, slice_name="held-in")

    agg = result.aggregate
    assert agg["accuracy"]["mean"] == pytest.approx(0.7)
    assert agg["macro_f1"]["mean"] == pytest.approx(0.694444, abs=1e-6)
    assert agg["kappa"]["mean"] == pytest.approx(0.545455, abs=1e-6)
    # Per-class F1 from the same fixture (0.6667 / 0.6667 / 0.75).
    pc = agg["per_class"]
    assert pc[S]["f1"]["mean"] == pytest.approx(0.666667, abs=1e-6)
    assert pc[C]["f1"]["mean"] == pytest.approx(0.666667, abs=1e-6)
    assert pc[N]["f1"]["mean"] == pytest.approx(0.75, abs=1e-6)


def test_repeats_mean_and_spread():
    """R=3 with one label varying across repeats → mean reported with non-zero spread."""
    items = [
        _t1("a", "fixed-alpha", S),
        _t1("b", "fixed-bravo", C),
        _t1("c", "fixed-charlie", N),
        _t1("v", "vary-delta", S),
    ]
    ground = {
        "fixed-alpha": S,
        "fixed-bravo": C,
        "fixed-charlie": N,
        # Consumed by call index across the 3 repeats: correct, correct, wrong.
        "vary-delta": [S, S, C],
    }
    provider = ScriptedProvider(ground=ground)
    result = run_tier1(items, n=1, repeats=3, provider=provider)

    acc = result.aggregate["accuracy"]
    assert result.repeats == 3
    assert acc["min"] < acc["max"], "expected a spread across repeats"
    assert acc["stdev"] > 0.0
    # repeat acc: 1.0, 1.0, 0.75 -> mean ~0.9167
    assert acc["mean"] == pytest.approx((1.0 + 1.0 + 0.75) / 3)


# --------------------------------------------------------------------------- #
# Tier 2 — τ sweep + the two §11 corner cases
# --------------------------------------------------------------------------- #


def test_tier2_tau_sweep():
    """At τ=1.0 a borderline (score 0.8) faithful case is a false positive; at τ=0.8 it isn't.

    So precision rises as τ drops while recall stays put — the trade-off the sweep exists
    to show.
    """
    a, dec_a, g_a = _t2_case("A", "alfa", 4, 5, gold_is_faithful=True, bucket="faithful_paraphrase")
    b, dec_b, g_b = _t2_case("B", "bravo", 3, 5, gold_is_faithful=False, bucket="mixed")
    c, dec_c, g_c = _t2_case("C", "charlie", 3, 3, gold_is_faithful=True, bucket="fully_grounded")

    provider = ScriptedProvider(
        decompose={**dec_a, **dec_b, **dec_c}, ground={**g_a, **g_b, **g_c}
    )
    result = run_tier2([a, b, c], n=1, repeats=1, provider=provider, taus=(1.0, 0.9, 0.8))
    agg = result.aggregate

    # tau=1.0: A (0.8) flagged -> false positive; precision 0.5, recall 1.0.
    assert agg["1"]["precision"]["mean"] == pytest.approx(0.5)
    assert agg["1"]["recall"]["mean"] == pytest.approx(1.0)
    # tau=0.8: A no longer flagged -> precision 1.0, recall still 1.0.
    assert agg["0.8"]["precision"]["mean"] == pytest.approx(1.0)
    assert agg["0.8"]["recall"]["mean"] == pytest.approx(1.0)
    # The sweep moves precision in the expected direction.
    assert agg["1"]["precision"]["mean"] < agg["0.8"]["precision"]["mean"]


def test_tier2_contradiction_flags_regardless_of_tau():
    """Corner case (a): n_contradicted>0 flags the answer at EVERY τ (load-bearing < 1.0)."""
    report = _report([_claim(S) for _ in range(4)] + [_claim(C)])
    assert report.n_contradicted == 1
    assert report.faithfulness_score == pytest.approx(0.8)
    for tau in (1.0, 0.9, 0.8):
        assert predicted_unfaithful(report, tau) is True
    # At τ=0.8 the score clause alone would NOT flag it (0.8 < 0.8 is False) — proving the
    # contradiction clause is what catches it. That is the load-bearing case.
    assert (report.faithfulness_score < 0.8) is False


def test_tier2_zero_claim_is_faithful():
    """Corner case (b): a score-is-None (0-claim) report is predicted FAITHFUL at every τ."""
    report = _report([])
    assert report.faithfulness_score is None
    for tau in (1.0, 0.9, 0.8):
        assert predicted_unfaithful(report, tau) is False


# --------------------------------------------------------------------------- #
# Held-in vs frozen split + persistence + refusal surfacing
# --------------------------------------------------------------------------- #


def test_held_in_vs_frozen_split(tmp_path):
    """Items split by `frozen`; frozen slice reports accuracy+κ only (no per-class/macro headline)."""
    items = [
        _t1("h1", "held-alpha", S, frozen=False),
        _t1("h2", "held-bravo", C, frozen=False),
        _t1("h3", "held-charlie", N, frozen=False),
        _t1("f1", "frozen-alpha", S, frozen=True),
        _t1("f2", "frozen-bravo", C, frozen=True),
        _t1("f3", "frozen-charlie", N, frozen=True),
    ]
    ground = {
        "held-alpha": S, "held-bravo": C, "held-charlie": N,
        "frozen-alpha": S, "frozen-bravo": C, "frozen-charlie": N,
    }
    provider = ScriptedProvider(ground=ground)
    ds = LoadedDataset(kind="tier1", items=items)

    summary, _lines, _path = run(
        tiers=["1"], n=1, repeats=1, slice_which="all",
        out_path=tmp_path / "hf.jsonl", provider=provider, mock_mode=False, datasets={"1": ds},
    )
    t1 = summary["tiers"]["1"]
    assert set(t1) == {"held-in", "frozen"}

    held = t1["held-in"]
    assert "macro_f1" in held and "per_class" in held

    frozen = t1["frozen"]
    assert "accuracy" in frozen and "kappa" in frozen
    assert "macro_f1" not in frozen, "frozen slice must not headline macro-F1 (§11 size honesty)"
    assert "per_class" not in frozen, "frozen slice must not headline per-class F1 (§11 size honesty)"
    assert "caption" in frozen and "accuracy" in frozen["caption"]


def test_run_jsonl_persistence(tmp_path):
    """A run writes a proper header + per-item lines + summary, and re-parses as JSONL."""
    t1_items = [_t1("t1-a", "p-alpha", S), _t1("t1-b", "p-bravo", C)]
    g1 = {"p-alpha": S, "p-bravo": C}

    a, dec_a, g_a = _t2_case("t2-a", "alfa", 2, 3, gold_is_faithful=False, bucket="mixed")
    b, dec_b, g_b = _t2_case("t2-b", "bravo", 3, 3, gold_is_faithful=True, bucket="fully_grounded")

    provider = ScriptedProvider(
        decompose={**dec_a, **dec_b}, ground={**g1, **g_a, **g_b}
    )
    out = tmp_path / "run.jsonl"
    summary, lines, path = run(
        tiers=["1", "2"], n=1, repeats=1, slice_which="all", out_path=out,
        provider=provider, mock_mode=False,
        datasets={
            "1": LoadedDataset(kind="tier1", items=t1_items),
            "2": LoadedDataset(kind="tier2", items=[a, b]),
        },
    )

    assert path == out and out.exists()
    parsed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    header = parsed[0]
    assert header["kind"] == "run_header"
    assert header["prompt_version"] == PROMPT_VERSION
    assert header["decompose_model"] == "claude-sonnet-4-6"
    assert header["ground_model"] == "claude-opus-4-8"
    for key in ("n", "repeats", "mock_mode"):
        assert key in header

    predictions = [line for line in parsed if line.get("kind") == "prediction"]
    assert predictions, "expected per-item prediction lines"
    assert any(p["tier"] == "1" for p in predictions)
    assert any(p["tier"] == "2" for p in predictions)

    summaries = [line for line in parsed if line.get("kind") == "summary"]
    assert len(summaries) == 1
    assert parsed[-1]["kind"] == "summary"


def test_refusal_affected_surfaced(tmp_path):
    """A scripted refusal in a Tier-2 case surfaces as summary `refusal_affected >= 1` (§7/§11)."""
    answer = "Scripted answer refuse body text."
    dec = Decomposition(
        claims=[
            DecomposedClaim(claim="Scripted refuse-trigger claim.", source_sentence="x"),
            DecomposedClaim(claim="Scripted normal-extra claim.", source_sentence="y"),
        ]
    )
    provider = ScriptedProvider(
        decompose={answer: dec},
        ground={"refuse-trigger": REFUSAL, "normal-extra": S},
    )
    item = Tier2Item(
        id="t2-ref", topic="scripted", source="Scripted source paragraph.",
        answer=answer, gold_is_faithful=False, bucket="mixed",
    )
    summary, _lines, _path = run(
        tiers=["2"], n=1, repeats=1, slice_which="held-in",
        out_path=tmp_path / "ref.jsonl", provider=provider, mock_mode=False,
        datasets={"2": LoadedDataset(kind="tier2", items=[item])},
    )
    assert summary["refusal_affected"] >= 1


def test_malformed_item_counted_as_skipped(tmp_path):
    """A malformed dataset item is logged + skipped + counted in the summary, not crashing."""
    good = _t1("t1-ok", "good-alpha", S)
    ds = LoadedDataset(
        kind="tier1",
        items=[good],
        skipped=[({"id": "t1-bad"}, "schema invalid: gold_label")],
    )
    provider = ScriptedProvider(ground={"good-alpha": S})
    summary, _lines, _path = run(
        tiers=["1"], n=1, repeats=1, slice_which="held-in",
        out_path=tmp_path / "sk.jsonl", provider=provider, mock_mode=False, datasets={"1": ds},
    )
    assert summary["skipped"] == 1


# --------------------------------------------------------------------------- #
# @api — real-provider quick smoke (skipped without a key)
# --------------------------------------------------------------------------- #


@pytest.mark.api
def test_real_quick_eval_smoke(real_provider, tmp_path):
    """With a key, a real --quick run completes and produces finite metrics (loose bounds)."""
    summary, _lines, _path = run(
        tiers=["1", "2"], n=1, repeats=1, slice_which="all",
        out_path=tmp_path / "api.jsonl", provider=real_provider, mock_mode=False,
    )
    t1 = summary["tiers"]["1"]["held-in"]
    assert 0.0 <= t1["macro_f1"]["mean"] <= 1.0
    assert -1.0 <= t1["kappa"]["mean"] <= 1.0
    t2 = summary["tiers"]["2"]["held-in"]
    assert 0.0 <= t2["tau_sweep"]["1"]["precision"]["mean"] <= 1.0
