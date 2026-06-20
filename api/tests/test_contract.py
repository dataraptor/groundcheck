"""Cross-layer contract test (Split 11, Deliverable 4) — no browser, no key.

Freezes the API↔frontend contract: the ``POST /check`` JSON (mock mode) must carry
**every** field the dc-html ``<script>`` reads, so a future engine change cannot
silently drop a field and break the page. There are two complementary guards:

1. An **explicit** expected-keys list, tied by name to Split 09's ``mapReport`` /
   ``renderVals`` (the only place the frontend reads the report). If the engine ever
   stops emitting one of these, the test fails with the exact missing key.
2. An **auto-derived** guard that parses ``app/GroundCheck.dc.html`` for every
   ``report.<field>`` it reads (and every raw claim field read inside ``mapReport``)
   and asserts each is present in the live response. This catches a *new* frontend
   read that the explicit list forgot, as well as engine drift.

If these two ever disagree, the contract has drifted — which is exactly the signal
this test exists to raise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from groundcheck_api.main import app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "core" / "examples"
_HTML = _REPO_ROOT / "app" / "GroundCheck.dc.html"

HALLUCINATED = json.loads((_EXAMPLES_DIR / "example_hallucinated.json").read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- #
# The explicit contract. These are the report fields Split 09's mapReport() and
# renderVals() read off the /check response. Keep this list in lock-step with the
# `<script type="text/x-dc">` block in app/GroundCheck.dc.html — if you add a
# report read there, add it here (the auto-derived guard below will remind you).
# --------------------------------------------------------------------------- #
REPORT_LEVEL_KEYS = {
    "faithfulness_score",  # renderVals: scoreText
    "n_claims",            # listed in the Split-11 brief; claims.length agrees with it
    "n_supported",         # renderVals: countsLine / liveMsg
    "n_not_enough_info",   # renderVals: liveMsg
    "n_contradicted",      # renderVals: meter/counts
    "n_low_confidence",    # renderVals: "to review"
    "n_refused",           # renderVals: refusalAffected → "* declined"
    "cost_usd",            # renderVals: footerLine
    "latency_s",           # renderVals: footerLine
    "n_runs",              # mapReport (votes runnerUp) + footerLine
    "prompt_version",      # renderVals: footerLine
    "warnings",            # renderVals: hasWarnings / oversize note
    "unlocated_sentences", # renderVals: hasUnlocated footnote
}

PER_CLAIM_KEYS = {
    "claim",            # mapReport → claim
    "label",            # mapReport → label (palette lookup)
    "supporting_span",  # mapReport → span
    "rationale",        # mapReport → rationale
    "votes",            # mapReport → "w · r" string
    "confidence",       # mapReport → confidence (borderline / review)
    "refused",          # mapReport → refused (declined treatment)
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_LLM", "mock")
    return TestClient(app)


@pytest.fixture(scope="module")
def script() -> str:
    html = _HTML.read_text(encoding="utf-8")
    m = re.search(r'<script type="text/x-dc"[^>]*>(.*?)</script>', html, re.DOTALL)
    assert m, "could not find the <script type=\"text/x-dc\"> block"
    return m.group(1)


def _body(client) -> dict:
    r = client.post("/check", json=HALLUCINATED)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# 1. Explicit contract — every listed field is present (report + per claim)
# --------------------------------------------------------------------------- #
def test_contract_report_matches_frontend_reads(client):
    body = _body(client)
    missing = REPORT_LEVEL_KEYS - set(body)
    assert not missing, f"response is missing report fields the frontend reads: {missing}"

    assert body["claims"], "expected at least one claim to validate the per-claim contract"
    for c in body["claims"]:
        claim_missing = PER_CLAIM_KEYS - set(c)
        assert not claim_missing, f"claim is missing fields the frontend reads: {claim_missing}"


def test_contract_field_types_match_frontend_assumptions(client):
    """The frontend does arithmetic / string ops on these — pin their JSON types."""
    body = _body(client)
    assert isinstance(body["faithfulness_score"], (int, float))  # mock → 0.625
    assert isinstance(body["n_runs"], int)
    assert isinstance(body["warnings"], list)
    assert isinstance(body["unlocated_sentences"], list)
    assert isinstance(body["cost_usd"], (int, float))
    assert isinstance(body["latency_s"], (int, float))
    c = body["claims"][0]
    assert isinstance(c["votes"], dict)        # mapReport indexes votes[label]
    assert isinstance(c["confidence"], (int, float))
    assert isinstance(c["refused"], bool)
    assert isinstance(c["label"], str)


def test_contract_would_fail_if_a_field_were_dropped(client):
    """Demonstrate the guard has teeth: a response with a key removed must be caught
    by the same membership check the contract test uses (so a real engine drop fails).
    """
    body = _body(client)
    for dropped in ("faithfulness_score", "n_refused", "warnings"):
        mutated = {k: v for k, v in body.items() if k != dropped}
        assert (REPORT_LEVEL_KEYS - set(mutated)) == {dropped}


# --------------------------------------------------------------------------- #
# 2. Auto-derived contract — parse the frontend for the fields it actually reads
# --------------------------------------------------------------------------- #
def test_every_report_field_read_by_script_is_in_response(client, script):
    """Any `report.<field>` the frontend reads must exist in the /check response.

    This is the drift guard: it does not depend on the explicit list above, so a new
    frontend read (or an engine rename) is caught automatically.
    """
    body = _body(client)
    read = set(re.findall(r"\breport\.([a-z_]+)", script))
    # `claims` is read as report.claims; it is the array the per-claim guard covers.
    read.discard("claims")
    missing = read - set(body)
    assert not missing, (
        f"the script reads report fields absent from the response: {missing}"
    )


def test_every_claim_field_read_in_mapreport_is_in_response(client, script):
    """Every raw claim field read inside mapReport() must exist on each claim.

    mapReport is the one place the frontend touches *raw* report claims (everything
    after it uses the mapped shape), so restricting the parse to its body gives the
    engine-side claim names without the post-map renames (e.g. span).
    """
    body = _body(client)
    m = re.search(r"mapReport\s*\(report\)\s*\{(.*?)\n  \}", script, re.DOTALL)
    assert m, "could not isolate the mapReport() body"
    read = set(re.findall(r"\bc\.([a-z_]+)", m.group(1)))
    assert read, "expected mapReport to read at least one claim field"
    for c in body["claims"]:
        missing = read - set(c)
        assert not missing, f"mapReport reads claim fields absent from the response: {missing}"


def test_explicit_and_derived_contracts_agree(script):
    """The hand-maintained list must cover everything the script actually reads."""
    report_read = set(re.findall(r"\breport\.([a-z_]+)", script))
    report_read.discard("claims")
    uncovered = report_read - REPORT_LEVEL_KEYS
    assert not uncovered, (
        f"the script reads report fields not in REPORT_LEVEL_KEYS — update the list: {uncovered}"
    )
    m = re.search(r"mapReport\s*\(report\)\s*\{(.*?)\n  \}", script, re.DOTALL)
    claim_read = set(re.findall(r"\bc\.([a-z_]+)", m.group(1)))
    uncovered_claim = claim_read - PER_CLAIM_KEYS
    assert not uncovered_claim, (
        f"mapReport reads claim fields not in PER_CLAIM_KEYS — update the list: {uncovered_claim}"
    )
