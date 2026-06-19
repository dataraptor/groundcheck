"""Shared eval-test fixtures — chiefly the :class:`ScriptedProvider` seam.

The harness math is verified with **no key and no network** by injecting a
deterministic provider that returns predetermined verdicts/decompositions per input.
That lets a test script the exact confusion matrix the harness should see, then
assert the harness reports the Split-06 hand-worked metrics — proving the harness
wires gold↔pred correctly, independent of any real model.

It also loads the repo-root ``.env`` for the optional ``@api`` smoke (same pattern as
``core/tests/conftest.py``), skipping when no real key is configured.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import pytest

from groundcheck.llm import ParseResult, Usage
from groundcheck.models import Decomposition, GroundingVerdict

_MOCK_USAGE = Usage(input_tokens=10, output_tokens=10)

# eval/tests/conftest.py -> parents[2] is the repo root, where .env lives.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def pytest_configure(config):
    """Register the ``@api`` marker (the eval layer has no pyproject ini section)."""
    config.addinivalue_line(
        "markers", "api: needs a real LLM key (Anthropic or Azure OpenAI); skipped without one"
    )


def _norm(text: str) -> str:
    """Lowercase + whitespace-collapsed form for substring key matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


# A sentinel label that scripts a refusal (rather than a real GroundingVerdict).
REFUSAL = "REFUSAL"


class ScriptedProvider:
    """A deterministic, no-network provider for exact harness-metric assertions.

    Construct with two substring-keyed scripts:

    * ``ground`` — ``{key: label}`` where ``label`` is a string
      (``SUPPORTED``/``CONTRADICTED``/``NOT_ENOUGH_INFO``), the :data:`REFUSAL`
      sentinel, or a **list** of those (an ordered sequence consumed by per-key call
      index, so a claim's label can change across repeats). The first key that is a
      substring of the normalized grounding text wins.
    * ``decompose`` — ``{key: Decomposition}``; the first key that is a substring of
      the normalized decompose text wins.

    Unmatched grounding → ``NOT_ENOUGH_INFO``; unmatched decompose → empty
    ``Decomposition`` (0 claims). Thread-safe (Split-05's pool grounds claims
    concurrently), so the per-key sequence index is lock-guarded.
    """

    def __init__(
        self,
        *,
        ground: Optional[dict[str, Any]] = None,
        decompose: Optional[dict[str, Decomposition]] = None,
    ) -> None:
        self._ground = {_norm(k): v for k, v in (ground or {}).items()}
        self._decompose = {_norm(k): v for k, v in (decompose or {}).items()}
        self._seq_index: dict[str, int] = {}
        self._lock = threading.Lock()

    def _next_label(self, key: str, value: Any) -> str:
        if not isinstance(value, list):
            return value
        with self._lock:
            idx = self._seq_index.get(key, 0)
            self._seq_index[key] = idx + 1
        return value[idx % len(value)]

    def parse(
        self,
        *,
        model: str,
        system: str,
        user_blocks: list[dict],
        output_model: type,
        max_tokens: int,
    ) -> ParseResult:
        text = _norm("".join(b.get("text", "") for b in user_blocks))

        if output_model is Decomposition:
            for key, dec in self._decompose.items():
                if key in text:
                    return _ok(dec)
            return _ok(Decomposition(claims=[]))

        if output_model is GroundingVerdict:
            for key, value in self._ground.items():
                if key not in text:
                    continue
                label = self._next_label(key, value)
                if label == REFUSAL:
                    return ParseResult(
                        parsed=None,
                        usage=_MOCK_USAGE,
                        stop_reason="refusal",
                        refused=True,
                        truncated=False,
                    )
                span = "" if label == "NOT_ENOUGH_INFO" else "scripted supporting span"
                return _ok(
                    GroundingVerdict(label=label, supporting_span=span, rationale="scripted")
                )
            return _ok(
                GroundingVerdict(
                    label="NOT_ENOUGH_INFO", supporting_span="", rationale="scripted default"
                )
            )

        raise AssertionError(f"ScriptedProvider got an unexpected output_model: {output_model!r}")


def _ok(parsed) -> ParseResult:
    return ParseResult(
        parsed=parsed, usage=_MOCK_USAGE, stop_reason="end_turn", refused=False, truncated=False
    )


# --------------------------------------------------------------------------- #
# Optional live provider for the @api smoke
# --------------------------------------------------------------------------- #


def _load_dotenv() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture
def real_provider():
    """A live LLM provider for the ``@api`` smoke, or skip if no real key is set."""
    _load_dotenv()
    from groundcheck.llm import get_provider

    if os.getenv("ANTHROPIC_API_KEY"):
        return get_provider("anthropic")
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return get_provider("openai")
    pytest.skip("no real LLM key configured (.env absent) — skipping @api smoke")
