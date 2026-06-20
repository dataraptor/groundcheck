"""Shared fixtures for the Split-11 browser e2e suite (no key for the gate).

What this provides:

* ``mock_server`` — a session-scoped uvicorn subprocess in **mock mode** (no key),
  serving the API + the static app same-origin; yields its base URL.
* ``nokey_server`` — a uvicorn subprocess forced onto a **real** provider with **no
  credentials**, so ``POST /check`` returns 503 ``missing_api_key`` (the missing-key
  state). Started lazily (only the missing-key test asks for it).
* ``open_app`` — navigate to the page and wait for the dc-runtime to mount; if it
  never mounts (the page needs ``unpkg.com`` for React/Babel), **skip** with a clear
  message instead of failing opaquely.

The suite is key-free but **not** network-free (the CDN React load). When chromium is
absent the whole suite skips cleanly; the no-key gate never depends on it.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The page the static mount serves (same path the API's "/" redirects to).
APP_PATH = "/app/GroundCheck.dc.html"

# How long to wait for the dc-runtime to fetch React/Babel from the CDN and mount.
MOUNT_TIMEOUT_MS = 30_000


# --------------------------------------------------------------------------- #
# Markers + browser availability
# --------------------------------------------------------------------------- #
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "e2e: browser end-to-end test (needs chromium + network for the CDN)"
    )
    config.addinivalue_line(
        "markers", "api: needs a real LLM key (skipped without one)"
    )


def _chromium_available() -> bool:
    """True only if Playwright's chromium build is actually installed on disk."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            path = p.chromium.executable_path
        return bool(path) and Path(path).exists()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip every e2e test (cleanly, with a message) when chromium isn't installed."""
    if _chromium_available():
        return
    skip = pytest.mark.skip(
        reason="chromium not installed — run `python -m playwright install chromium` "
        "to enable the e2e suite (the no-key gate does not depend on it)."
    )
    for item in items:
        item.add_marker(skip)


# --------------------------------------------------------------------------- #
# uvicorn subprocess helpers
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:  # not up yet
            last_err = exc
            time.sleep(0.25)
    raise RuntimeError(f"uvicorn at {base_url} never became healthy: {last_err}")


def _start_uvicorn(env_overrides: dict[str, str], strip_keys: bool = False) -> tuple[subprocess.Popen, str]:
    """Start ``groundcheck_api`` on a free port; return (process, base_url)."""
    port = _free_port()
    env = dict(os.environ)
    if strip_keys:
        for k in ("GROUNDCHECK_LLM", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
            env.pop(k, None)
    env.update(env_overrides)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "groundcheck_api.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url)
    except Exception:
        proc.terminate()
        raise
    return proc, base_url


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover — defensive
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def mock_server() -> Iterator[str]:
    """A uvicorn server in mock mode (no key); serves the API + static app."""
    proc, base_url = _start_uvicorn({"GROUNDCHECK_LLM": "mock"})
    try:
        yield base_url
    finally:
        _stop(proc)


@pytest.fixture(scope="session")
def nokey_server() -> Iterator[str]:
    """A uvicorn server on a real provider with NO credentials → /check is 503."""
    # No GROUNDCHECK_LLM and no keys → get_provider() picks AnthropicProvider, which
    # raises a missing-key RuntimeError on /check (mapped to 503 missing_api_key).
    proc, base_url = _start_uvicorn({}, strip_keys=True)
    try:
        yield base_url
    finally:
        _stop(proc)


# --------------------------------------------------------------------------- #
# Page-mount helper (CDN-aware skip)
# --------------------------------------------------------------------------- #
@pytest.fixture
def open_app() -> Callable:
    """Return ``open_app(page, base_url)`` → navigates and waits for the app to mount.

    Mount = the "Check faithfulness" button is visible (the dc-runtime has loaded
    React/Babel from the CDN and rendered). On timeout we **skip** (no network for the
    CDN) rather than fail, surfacing a readable reason.
    """

    def _open(page, base_url: str):
        page.goto(base_url + APP_PATH, wait_until="domcontentloaded")
        try:
            page.get_by_role("button", name="Check faithfulness").wait_for(
                state="visible", timeout=MOUNT_TIMEOUT_MS
            )
        except Exception:
            pytest.skip(
                "the dc-runtime never mounted — the page needs network for the unpkg "
                "React/Babel CDN (see README 'Known wart'). Not a flaky test."
            )
        return page

    return _open
