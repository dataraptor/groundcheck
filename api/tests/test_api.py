"""API tests — the whole stack runs key-free via ``TestClient`` in mock mode.

The HTTP layer is a thin adapter, so these tests pin (a) the worked-example numbers
flow through unchanged, (b) the response JSON shape Split 09 maps against, and (c)
every engine edge case (empty answer / bad n / missing key / oversize) lands on a
clean status + body with **no** stack trace.

A ``@pytest.mark.api`` smoke calls a live model and is skipped without a key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from groundcheck_api.main import app

# api/tests/test_api.py -> parents[2] is the repo root, where core/examples lives.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "core" / "examples"


def _load_example(name: str) -> dict:
    return json.loads((_EXAMPLES_DIR / name).read_text(encoding="utf-8"))


HALLUCINATED = _load_example("example_hallucinated.json")
GROUNDED = _load_example("example_grounded.json")


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the engine forced into key-free mock mode."""
    monkeypatch.setenv("GROUNDCHECK_LLM", "mock")
    return TestClient(app)


# --------------------------------------------------------------------------- #
# POST /check — the money path
# --------------------------------------------------------------------------- #


def test_check_worked_example(client):
    """The §5 hallucinated example flows through the adapter as the canonical 62%."""
    r = client.post("/check", json=HALLUCINATED)
    assert r.status_code == 200
    body = r.json()
    assert body["faithfulness_score"] == 0.625
    assert body["n_supported"] == 5
    assert body["n_not_enough_info"] == 3
    assert body["n_claims"] == 8


def test_check_response_shape(client):
    """The response carries EVERY field Split 09 maps (report + warnings, per claim)."""
    r = client.post("/check", json=HALLUCINATED)
    assert r.status_code == 200
    body = r.json()

    report_keys = {
        "claims",
        "n_claims",
        "n_supported",
        "n_contradicted",
        "n_not_enough_info",
        "n_low_confidence",
        "n_refused",
        "faithfulness_score",
        "cost_usd",
        "latency_s",
        "prompt_version",
        "n_runs",
        "warnings",
    }
    assert report_keys <= set(body), f"missing: {report_keys - set(body)}"
    # The Split-05 presentation fields ride along too (Splits 10/11 use them).
    assert "highlighted_html" in body
    assert "unlocated_sentences" in body

    claim_keys = {
        "claim",
        "source_sentence",
        "label",
        "supporting_span",
        "rationale",
        "votes",
        "confidence",
        "refused",
    }
    assert body["claims"], "expected at least one claim"
    for c in body["claims"]:
        assert claim_keys <= set(c), f"claim missing: {claim_keys - set(c)}"
    # votes is a dict, confidence a float, refused a bool — the UI relies on these.
    first = body["claims"][0]
    assert isinstance(first["votes"], dict)
    assert isinstance(first["confidence"], (int, float))
    assert isinstance(first["refused"], bool)


def test_check_grounded_example(client):
    """The fully-grounded example scores 1.0 with every claim SUPPORTED."""
    r = client.post("/check", json=GROUNDED)
    assert r.status_code == 200
    body = r.json()
    assert body["faithfulness_score"] == 1.0
    assert body["n_claims"] > 0
    assert body["n_supported"] == body["n_claims"]
    assert all(c["label"] == "SUPPORTED" for c in body["claims"])


def test_check_empty_answer_is_na(client):
    """An empty answer is VALID -> the N/A path (200, score None), not an error."""
    r = client.post("/check", json={"source": HALLUCINATED["source"], "answer": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["faithfulness_score"] is None
    assert body["n_claims"] == 0


def test_check_whitespace_answer_is_na(client):
    """Whitespace-only answers also route to N/A, never a crash."""
    r = client.post("/check", json={"source": HALLUCINATED["source"], "answer": "   \n\t  "})
    assert r.status_code == 200
    assert r.json()["faithfulness_score"] is None


@pytest.mark.parametrize("bad_n", [0, 6, -1, 100])
def test_check_bad_n(client, bad_n):
    """n outside [N_RUNS_MIN, N_RUNS_MAX] is rejected by validation with a 422."""
    r = client.post(
        "/check", json={"source": "s", "answer": "a", "n": bad_n}
    )
    assert r.status_code == 422


@pytest.mark.parametrize("good_n", [1, 3, 5])
def test_check_good_n_accepted(client, good_n):
    """The supported stepper range is accepted."""
    r = client.post(
        "/check", json={"source": HALLUCINATED["source"], "answer": HALLUCINATED["answer"], "n": good_n}
    )
    assert r.status_code == 200
    assert r.json()["n_runs"] == good_n


def test_check_missing_key_returns_503(monkeypatch):
    """Force the real provider with no key -> 503 missing_api_key, no traceback."""
    # Away from mock, and strip every real credential so the provider must error.
    monkeypatch.delenv("GROUNDCHECK_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    client = TestClient(app)

    r = client.post("/check", json={"source": "some source", "answer": "some answer"})
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "missing_api_key"
    assert isinstance(body["error"], str) and body["error"]
    # No stack trace leaks to the client (spec §17).
    blob = json.dumps(body)
    assert "Traceback" not in blob
    assert 'File "' not in blob


# --------------------------------------------------------------------------- #
# GET /health, GET /examples
# --------------------------------------------------------------------------- #


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mock_mode"] is True
    assert body["prompt_version"] == "v3"
    assert body["models"] == {
        "decompose": "claude-sonnet-4-6",
        "ground": "claude-opus-4-8",
    }


def test_health_never_needs_a_key(monkeypatch):
    """Health works even with no provider configured (it never touches the network)."""
    monkeypatch.delenv("GROUNDCHECK_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["mock_mode"] is False


def test_examples(client):
    r = client.get("/examples")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2
    ids = {it["id"] for it in items}
    assert ids == {"hallucinated", "grounded"}
    for it in items:
        assert it["source"].strip()
        assert it["answer"].strip()
    halluc = next(it for it in items if it["id"] == "hallucinated")
    assert halluc["source"] == HALLUCINATED["source"]
    assert halluc["answer"] == HALLUCINATED["answer"]


# --------------------------------------------------------------------------- #
# Static app (same-origin)
# --------------------------------------------------------------------------- #


def test_static_app_served(client):
    r_html = client.get("/app/GroundCheck.dc.html")
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers["content-type"]

    r_js = client.get("/app/support.js")
    assert r_js.status_code == 200

    r_root = client.get("/", follow_redirects=False)
    assert r_root.status_code in (302, 307)
    assert r_root.headers["location"] == "/app/GroundCheck.dc.html"


# --------------------------------------------------------------------------- #
# Oversize input -> truncation surfaced (spec §17)
# --------------------------------------------------------------------------- #


def test_oversize_warning_surfaced(client):
    """An over-cap answer is truncated WITH a surfaced warning, not silently."""
    # MAX_ANSWER_TOKENS (16000) * 4 chars/token = 64000-char cap; go well past it.
    oversize_answer = "This is a filler sentence. " * 4000
    assert len(oversize_answer) > 64000
    r = client.post("/check", json={"source": HALLUCINATED["source"], "answer": oversize_answer})
    assert r.status_code == 200
    body = r.json()
    assert body["warnings"], "expected a truncation warning to be surfaced"
    assert any("cap" in w.lower() or "truncat" in w.lower() for w in body["warnings"])
    # Still a valid report (the N/A or scored shape, never a crash).
    assert "faithfulness_score" in body
    assert "n_claims" in body


def test_oversize_source_warning_surfaced(client):
    """An over-cap SOURCE is truncated WITH a surfaced warning (Split 11 §17 chain).

    The source cap (``MAX_SOURCE_TOKENS``) was defined but un-enforced before Split 11
    (see PROGRESS bug provenance); the engine now caps it in ``check()`` and the API
    surfaces the warning the same way it does for the answer cap.
    """
    oversize_source = "Long source sentence. " * 4000  # > 64000-char cap
    assert len(oversize_source) > 64000
    r = client.post("/check", json={"source": oversize_source, "answer": HALLUCINATED["answer"]})
    assert r.status_code == 200
    body = r.json()
    assert body["warnings"], "expected a source-truncation warning to be surfaced"
    assert any("source" in w.lower() and "cap" in w.lower() for w in body["warnings"])
    assert "faithfulness_score" in body


# --------------------------------------------------------------------------- #
# Live smoke (opt-in; needs a real key)
# --------------------------------------------------------------------------- #


@pytest.mark.api
def test_check_real_smoke():
    """With a real key, the hallucinated example scores mostly-but-not-fully faithful."""
    _load_dotenv()
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        pytest.skip("no real LLM key configured (.env absent) — skipping @api smoke")

    client = TestClient(app)
    r = client.post("/check", json={**HALLUCINATED, "n": 3})
    assert r.status_code == 200
    body = r.json()
    # Loose band: real models split compounds harder than the pinned mock 8 claims
    # (see PROGRESS Split 03/05 divergences), so widen n_claims and the score band.
    assert body["faithfulness_score"] is not None
    assert 0.4 <= body["faithfulness_score"] <= 0.8
    assert body["n_claims"] >= 6


def _load_dotenv() -> None:
    """Load repo-root .env (Azure gpt-5.5 creds) for the opt-in @api smoke only."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
