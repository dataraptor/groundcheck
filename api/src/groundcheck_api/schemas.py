"""Request / response contracts for the HTTP layer (Split 08).

These are deliberately thin. The response simply **mirrors** the engine's
:class:`~groundcheck.models.FaithfulnessReport` (so the frontend in Split 09 maps
fields 1:1) and adds one sibling key, ``warnings`` — surfaced engine notices such as
oversize-truncation (spec §17: never silently truncate). Keeping :class:`CheckResponse`
a *subclass* of the report guarantees it carries every report field automatically; a
test pins the exact JSON shape so Split 09's mapping cannot drift.
"""

from __future__ import annotations

from typing import Optional

from groundcheck.config import DEFAULT_N_RUNS, N_RUNS_MAX, N_RUNS_MIN
from groundcheck.models import FaithfulnessReport
from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """A faithfulness-check request: an answer to verify against a source.

    An empty / whitespace ``answer`` is **valid** — it routes to the N/A path
    (``faithfulness_score == None``), not a validation error. ``n`` is bounded to the
    supported stepper range; out-of-range values are rejected by pydantic with a 422
    before the route body runs.
    """

    source: str
    answer: str
    n: int = Field(
        default=DEFAULT_N_RUNS,
        ge=N_RUNS_MIN,
        le=N_RUNS_MAX,
        description=f"grounding runs per claim (majority vote); {N_RUNS_MIN}–{N_RUNS_MAX}.",
    )


class CheckResponse(FaithfulnessReport):
    """The full faithfulness report plus surfaced engine warnings.

    Inherits every :class:`~groundcheck.models.FaithfulnessReport` field (claims,
    counts, score|None, cost/latency/prompt_version/n_runs, highlighted_html,
    unlocated_sentences) and adds ``warnings``. Build it with :meth:`from_report`.
    """

    warnings: list[str] = Field(
        default_factory=list,
        description="engine notices surfaced to the user (e.g. oversize truncation, spec §17).",
    )

    @classmethod
    def from_report(
        cls, report: FaithfulnessReport, warnings: Optional[list[str]] = None
    ) -> "CheckResponse":
        """Wrap an engine report, attaching captured warnings (no business logic)."""
        return cls(**report.model_dump(), warnings=list(warnings or []))


class ErrorResponse(BaseModel):
    """A clean error body — never a stack trace (spec §17).

    ``code`` is a stable machine string the frontend switches on
    (``missing_api_key`` / ``engine_error``); ``error`` is a human message; ``detail``
    is an optional remediation hint.
    """

    code: str
    error: str
    detail: Optional[str] = None
