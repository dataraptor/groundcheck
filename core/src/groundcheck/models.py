"""Data contracts for GroundCheck (spec §7).

Two categories of model live here:

1. **Sent to the API** — the structured-output contracts (`DecomposedClaim`,
   `Decomposition`, `GroundingVerdict`). These are kept deliberately simple: only
   `str`/`list`/enum fields, no `minLength`/`maxLength`/`minimum`/`maximum`/
   `multipleOf` and no recursion, so the JSON schema is compatible with Anthropic
   structured outputs (README "API facts").

2. **Assembled in code** — the result types (`ClaimResult`, `FaithfulnessReport`).
   These are never sent to the model; they are built by the pipeline (Split 05) and
   must serialize cleanly to JSON for the CLI `--json` flag.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- the three labels (spec §7, Appendix A) ---
Label = Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]


# --------------------------------------------------------------------------- #
# (a) Sent to the API — structured-output contracts (keep simple, no bounds).
# --------------------------------------------------------------------------- #
class DecomposedClaim(BaseModel):
    """One atomic factual claim extracted from the answer."""

    claim: str
    source_sentence: str  # verbatim sentence from the ANSWER this claim came from


class Decomposition(BaseModel):
    """The full set of atomic claims for one answer."""

    claims: list[DecomposedClaim]


class GroundingVerdict(BaseModel):
    """The judge's verdict for a single claim against the source."""

    label: Label
    supporting_span: str  # verbatim SOURCE span for SUPPORTED/CONTRADICTED; "" for NEI
    rationale: str  # one sentence, grounded in the source


# --------------------------------------------------------------------------- #
# (b) Assembled in code — result types (not sent to the API).
# --------------------------------------------------------------------------- #
class ClaimResult(BaseModel):
    """A claim's grounding result after N-run majority voting (spec §7, §9)."""

    claim: str
    source_sentence: str
    label: Label
    supporting_span: str
    rationale: str
    votes: dict[str, int]  # e.g. {"SUPPORTED": 2, "NOT_ENOUGH_INFO": 1}
    confidence: float  # max_votes / n_runs (3-0 -> 1.0; 2-1 -> 0.67)
    refused: bool = False  # the judge declined on at least one run (spec §17)


class FaithfulnessReport(BaseModel):
    """The full faithfulness report for one check (spec §7)."""

    claims: list[ClaimResult]
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_not_enough_info: int
    n_low_confidence: int  # claims with confidence < 1.0
    n_refused: int  # claims where the judge declined -> counted as NEI (spec §7/§17)
    faithfulness_score: Optional[float] = Field(
        description="SUPPORTED / n_claims. None ⇒ N/A (no checkable claims)."
    )
    cost_usd: float
    latency_s: float
    prompt_version: str
    n_runs: int
    # --- presentation fields, set by the pipeline AFTER from_claims (Split 05) ----- #
    # Both are optional with the defaults below so the Split-01 ``from_claims`` path
    # (and its tests) still build a valid report; the pipeline fills them in via
    # ``highlight_answer`` and they serialize into the API/UI response. See PROGRESS
    # "Open divergences" — these are required by Splits 08/10/11.
    highlighted_html: str = ""  # answer with worst-verdict <span> highlighting (spec §8)
    unlocated_sentences: list[str] = Field(
        default_factory=list,
        description="source_sentences the highlighter could not locate in the answer (§8 give-up).",
    )

    @classmethod
    def from_claims(
        cls,
        claims: list[ClaimResult],
        *,
        cost_usd: float,
        latency_s: float,
        n_runs: int,
        prompt_version: str,
    ) -> "FaithfulnessReport":
        """Derive all counts and the score from a list of claim results.

        Pure (no I/O); reused by the pipeline (Split 05).

        The score is ``n_supported / n_claims`` and is **None** when there are no
        claims — reported as "N/A (no checkable claims)", never a misleading 100%
        (spec §7, §19-1).
        """
        n_claims = len(claims)
        n_supported = sum(1 for c in claims if c.label == "SUPPORTED")
        n_contradicted = sum(1 for c in claims if c.label == "CONTRADICTED")
        n_not_enough_info = sum(1 for c in claims if c.label == "NOT_ENOUGH_INFO")
        n_low_confidence = sum(1 for c in claims if c.confidence < 1.0)
        n_refused = sum(1 for c in claims if c.refused)

        faithfulness_score = (n_supported / n_claims) if n_claims else None

        return cls(
            claims=claims,
            n_claims=n_claims,
            n_supported=n_supported,
            n_contradicted=n_contradicted,
            n_not_enough_info=n_not_enough_info,
            n_low_confidence=n_low_confidence,
            n_refused=n_refused,
            faithfulness_score=faithfulness_score,
            cost_usd=cost_usd,
            latency_s=latency_s,
            n_runs=n_runs,
            prompt_version=prompt_version,
        )
