"""FastAPI service: a thin HTTP adapter over ``groundcheck.check`` (spec §4, Split 08).

Routes
------
* ``POST /check``   — validate → ``check()`` → capture warnings → map errors → return.
* ``GET  /examples``— the two worked examples, for the frontend to prefill.
* ``GET  /health``  — liveness + prompt/model identity; never touches the network.
* ``GET  /``        — redirect to the static app.
* ``/app/*``        — the static front-end (``GroundCheck.dc.html`` + ``support.js``),
  served **same-origin** so the page can ``fetch('/check')`` with no CORS.

Discipline: this module contains **no** grounding / scoring / highlighting logic. It
only adapts HTTP to the in-process engine and maps every engine edge case (missing
key, oversize input, refusal) to a clean JSON body — never a stack trace (spec §17).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import groundcheck
from groundcheck.config import DECOMPOSE_MODEL, GROUND_MODEL
from groundcheck.prompts import PROMPT_VERSION
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .schemas import CheckRequest, CheckResponse, ErrorResponse

logger = logging.getLogger("groundcheck_api")

# --------------------------------------------------------------------------- #
# Repo-relative paths (resolved from this file, NOT the CWD, so uvicorn works
# regardless of where it is launched). Both are overridable by env var.
#   api/src/groundcheck_api/main.py  ->  parents[3] == repo root.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(os.getenv("GROUNDCHECK_APP_DIR") or (_REPO_ROOT / "app"))
EXAMPLES_DIR = Path(os.getenv("GROUNDCHECK_EXAMPLES_DIR") or (_REPO_ROOT / "core" / "examples"))

# The worked examples (Split 05). Filename -> {id, name} the frontend prefills with.
_EXAMPLE_FILES = [
    ("example_hallucinated.json", "hallucinated", "Hallucinated answer (62%)"),
    ("example_grounded.json", "grounded", "Grounded answer (100%)"),
]


def _mock_mode() -> bool:
    """True when the demo runs key-free (``GROUNDCHECK_LLM=mock``)."""
    return os.getenv("GROUNDCHECK_LLM", "").strip().lower() == "mock"


# --------------------------------------------------------------------------- #
# Warning capture — surface engine notices (oversize truncation, etc.) emitted
# via ``logging.warning`` on the ``groundcheck`` logger during a check (spec §17).
# --------------------------------------------------------------------------- #


class _ListHandler(logging.Handler):
    """Collect the formatted message of every WARNING+ record into a list."""

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.append(record.getMessage())
        except Exception:  # pragma: no cover — a logging sink must never raise
            pass


@contextmanager
def _capture_warnings() -> Iterator[list[str]]:
    """Capture ``groundcheck`` WARNING+ messages emitted inside the ``with`` block.

    Attaches a temporary handler to the ``groundcheck`` logger (so it catches the
    decompose/ground/pipeline children via propagation) and lowers the logger's
    threshold to WARNING for the window if it was set higher — never raising it, so
    finer logs are not suppressed. Both are restored on exit.
    """
    captured: list[str] = []
    handler = _ListHandler(captured)
    gc_logger = logging.getLogger("groundcheck")
    original_level = gc_logger.level
    if gc_logger.getEffectiveLevel() > logging.WARNING:
        gc_logger.setLevel(logging.WARNING)
    gc_logger.addHandler(handler)
    try:
        yield captured
    finally:
        gc_logger.removeHandler(handler)
        gc_logger.setLevel(original_level)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="GroundCheck API",
    version="0.1.0",
    description="Thin HTTP adapter over the GroundCheck faithfulness engine.",
)

# CORS is NOT needed for the production path (the page is served same-origin under
# /app and fetches /check on the same host). This is a dev-only convenience for when
# the frontend is opened on a different localhost port — scoped to loopback origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/check",
    response_model=CheckResponse,
    responses={
        503: {"model": ErrorResponse, "description": "No API key configured."},
        502: {"model": ErrorResponse, "description": "The engine failed."},
    },
)
def post_check(req: CheckRequest):
    """Verify ``answer`` against ``source`` and return the full report + warnings.

    Pydantic rejects ``n`` outside ``[N_RUNS_MIN, N_RUNS_MAX]`` with a 422 before this
    body runs. An empty/whitespace answer is valid → the N/A report (200). Missing
    key → 503; any other engine/SDK failure → 502. No stack trace ever reaches the
    client.
    """
    try:
        with _capture_warnings() as warnings:
            # The ONLY business call. provider via get_provider() (honors mock).
            report = groundcheck.check(req.source, req.answer, n=req.n)
    except RuntimeError as exc:
        # The provider raises RuntimeError only for a missing key (AnthropicProvider
        # names ANTHROPIC_API_KEY; OpenAIProvider names the Azure vars) — a
        # server-config problem, not the client's fault.
        logger.warning("check failed: no API key configured (%s)", exc)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                code="missing_api_key",
                error=str(exc),
                detail="Set ANTHROPIC_API_KEY (or Azure OpenAI creds), "
                "or run with GROUNDCHECK_LLM=mock.",
            ).model_dump(),
        )
    except Exception as exc:  # noqa: BLE001 — map any engine/SDK error to a safe body
        # Log the full trace SERVER-SIDE only; the client gets a safe message.
        logger.exception("check failed with an unexpected engine error")
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(
                code="engine_error",
                error="The verification engine failed to complete the check.",
                detail=type(exc).__name__,
            ).model_dump(),
        )

    return CheckResponse.from_report(report, warnings)


@app.get("/examples")
def get_examples():
    """The two worked examples (Split 05) for the frontend to prefill.

    Loaded from ``core/examples/*.json`` via a repo-relative path (no hardcoded
    absolute path; override with ``GROUNDCHECK_EXAMPLES_DIR``).
    """
    import json

    examples = []
    for filename, example_id, name in _EXAMPLE_FILES:
        path = EXAMPLES_DIR / filename
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("example file missing: %s (skipping)", path)
            continue
        examples.append(
            {
                "id": example_id,
                "name": name,
                "source": data.get("source", ""),
                "answer": data.get("answer", ""),
            }
        )
    return examples


@app.get("/health")
def health():
    """Liveness + identity. Never touches the network or requires a key."""
    return {
        "status": "ok",
        "prompt_version": PROMPT_VERSION,
        "mock_mode": _mock_mode(),
        "models": {"decompose": DECOMPOSE_MODEL, "ground": GROUND_MODEL},
    }


@app.get("/", include_in_schema=False)
def root():
    """Send the bare origin to the static app."""
    return RedirectResponse(url="/app/GroundCheck.dc.html")


# Static app — mounted LAST so it never shadows the API routes above. Serving it
# here (same origin as /check) is what removes the need for CORS in production.
if APP_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(APP_DIR), html=True), name="app")
else:  # pragma: no cover — the repo ships app/, but fail loud rather than silently
    logger.warning("app directory not found at %s — static app not served", APP_DIR)
