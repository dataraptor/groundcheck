"""Split 12 — guard the root README against drift (no key).

The README's no-key quickstart promises that the documented CLI command prints
**62%**. This test parses that exact command out of `README.md`, runs it through
`cli.main`, and asserts the headline — so the README can't silently diverge from a
runnable command (or from the worked example). It also pins two honesty invariants:
the money-demo screenshot is referenced, and the banned "catches all hallucinations"
phrasing never appears (spec §11).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from groundcheck import cli

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_LLM", "mock")


def _readme_text() -> str:
    return _README.read_text(encoding="utf-8")


def test_readme_exists_and_references_screenshot():
    text = _readme_text()
    assert "docs/money-demo.png" in text, "README must embed the money-demo screenshot"
    assert (_REPO_ROOT / "docs" / "money-demo.png").exists(), "screenshot file missing"
    # The no-key path is the headline of the quickstart.
    assert "GROUNDCHECK_LLM=mock" in text


def test_readme_quickstart_cli_command_prints_62(capsys):
    """Run the *exact* mock CLI command documented in the README → asserts 62%."""
    text = _readme_text()
    # Find the documented `python -m groundcheck.cli check --example <path>` command.
    m = re.search(r"groundcheck\.cli\s+check\s+--example\s+(\S+\.json)", text)
    assert m, "README quickstart no longer documents the mock CLI `--example` command"
    example_path = (_REPO_ROOT / m.group(1)).resolve()
    assert example_path.exists(), f"README points at a missing example: {example_path}"

    rc = cli.main(["check", "--example", str(example_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "62%" in out, "the documented quickstart command must still print 62%"

    # The README states the same headline next to the command.
    assert "62%" in text


def test_readme_honesty_phrasing():
    """Spec §11: never *claim* the detector 'catches all hallucinations'.

    The phrase may appear only inside a negation (the README explicitly tells the
    reader NOT to claim it); it must never stand as a positive claim.
    """
    text = _readme_text().lower()
    phrase = "catches all hallucinations"
    for m in re.finditer(re.escape(phrase), text):
        window = text[max(0, m.start() - 40):m.start()]
        assert "never" in window, "the banned phrase must only appear in a 'never' negation"
    # The honest, traceable framing must be present.
    assert "macro-f1" in text and "frozen" in text
