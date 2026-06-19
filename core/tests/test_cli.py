"""Tests for Split 05 — the ``check`` CLI (no key; mock mode + clean error paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundcheck import cli

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_HALLUCINATED = str(_EXAMPLES / "example_hallucinated.json")
_GROUNDED = str(_EXAMPLES / "example_grounded.json")


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    """Default every CLI test to mock mode (no key) unless it overrides the env."""
    monkeypatch.setenv("GROUNDCHECK_LLM", "mock")


# --------------------------------------------------------------------------- #
# Mock-mode money demo prints 62%
# --------------------------------------------------------------------------- #


def test_cli_check_example_mock_prints_62(capsys):
    rc = cli.main(["check", "--example", _HALLUCINATED])
    out = capsys.readouterr().out
    assert rc == 0
    assert "62%" in out
    assert "5/8 grounded" in out
    # The two fabrications appear in the per-claim breakdown.
    assert "exactly 25%" in out
    assert "leading cause of death" in out


def test_cli_check_grounded_prints_100(capsys):
    rc = cli.main(["check", "--example", _GROUNDED])
    out = capsys.readouterr().out
    assert rc == 0
    assert "100%" in out


# --------------------------------------------------------------------------- #
# --json
# --------------------------------------------------------------------------- #


def test_cli_json_flag(capsys):
    rc = cli.main(["check", "--example", _HALLUCINATED, "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    report = json.loads(out)  # parses as JSON
    assert report["faithfulness_score"] == 0.625
    assert report["n_supported"] == 5
    assert report["n_claims"] == 8
    # The presentation fields serialize into the JSON (Split 08/11 contract).
    assert "highlighted_html" in report
    assert "unlocated_sentences" in report


# --------------------------------------------------------------------------- #
# Source/answer file inputs
# --------------------------------------------------------------------------- #


def test_cli_source_answer_files(tmp_path, capsys):
    data = json.loads(Path(_HALLUCINATED).read_text(encoding="utf-8"))
    src = tmp_path / "s.txt"
    ans = tmp_path / "a.txt"
    src.write_text(data["source"], encoding="utf-8")
    ans.write_text(data["answer"], encoding="utf-8")
    rc = cli.main(["check", "--source-file", str(src), "--answer-file", str(ans)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "62%" in out


# --------------------------------------------------------------------------- #
# N/A path through the CLI (empty answer)
# --------------------------------------------------------------------------- #


def test_cli_na_for_empty_answer(tmp_path, capsys):
    ex = tmp_path / "empty.json"
    ex.write_text(json.dumps({"source": "some source", "answer": "   "}), encoding="utf-8")
    rc = cli.main(["check", "--example", str(ex)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "N/A" in out


# --------------------------------------------------------------------------- #
# Missing-key: clean guidance, non-zero exit, no traceback
# --------------------------------------------------------------------------- #


def test_cli_missing_key_message(monkeypatch, capsys):
    # Select a real provider with no key configured.
    monkeypatch.setenv("GROUNDCHECK_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = cli.main(["check", "--example", _HALLUCINATED])
    captured = capsys.readouterr()
    assert rc != 0
    combined = captured.out + captured.err
    # Names the fix (set a key / use mock) and shows no Python traceback.
    assert "mock" in combined.lower()
    assert "Traceback" not in combined


# --------------------------------------------------------------------------- #
# Bad input → clean usage error (exit 2)
# --------------------------------------------------------------------------- #


def test_cli_missing_inputs_usage_error(capsys):
    rc = cli.main(["check"])  # neither --example nor the file pair
    err = capsys.readouterr().err
    assert rc == 2
    assert "provide --example" in err


def test_cli_missing_example_file(capsys):
    rc = cli.main(["check", "--example", "no_such_file.json"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "not found" in err.lower()
    assert "Traceback" not in err
