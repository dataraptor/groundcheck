"""Shared test fixtures.

The whole suite is key-free by default: real providers are exercised through
injected stub clients and everything else runs on :class:`MockProvider`. The one
exception is the ``@pytest.mark.api`` smoke tests, which want a *live* model. The
:func:`real_provider` fixture below loads the project ``.env`` (the Azure gpt-5.5
creds — the only real key in this environment, see ``tmp/split/PROGRESS.md``) and
returns a live provider, **skipping** the test when no real key is configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# core/tests/conftest.py → parents[2] is the repo root, where .env lives.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv() -> None:
    """Load ``repo-root/.env`` (``key=value`` lines) into ``os.environ`` if present.

    Uses ``setdefault`` so a value already exported in the shell always wins, and
    only runs from inside the :func:`real_provider` fixture so the rest of the
    (key-free) suite is unaffected.
    """
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
    """A live LLM provider for ``@api`` smokes, or skip if no real key is set."""
    _load_dotenv()
    from groundcheck.llm import get_provider

    if os.getenv("ANTHROPIC_API_KEY"):
        return get_provider("anthropic")
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return get_provider("openai")
    pytest.skip("no real LLM key configured (.env absent) — skipping @api smoke")
