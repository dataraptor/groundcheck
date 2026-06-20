"""The orchestrator — ``check(source, answer)`` (spec §4/§5/§10).

This ties the engine together: decompose the answer into atomic claims (Sonnet),
ground each claim N times against the source (Opus, concurrent), score, highlight,
and assemble a :class:`~groundcheck.models.FaithfulnessReport`. It is a pure
in-process function — the CLI, the API, and the eval harness all import it directly
(no service-to-service HTTP).

Two design points worth stating outright:

* **Honest cost (spec §10):** the per-check ``cost_usd`` includes the Sonnet
  decompose cost *plus* every Opus grounding-run cost — the decompose cost is never
  silently dropped.
* **Warm-one-first caching gate (spec §10):** within one check the judge sees the
  same SOURCE on every grounding call. A cache entry is only readable *after the
  first response begins streaming*, so for a source large enough to cache (≥ the
  4096-token Opus floor) we ground the **first claim synchronously** to warm the
  SOURCE cache, then fan the rest out concurrently. For small demo sources (no cache
  benefit at all) we fan out immediately. Either branch produces the same report;
  only the scheduling differs.

Robustness: grounding runs concurrently but the report's claims are assembled in
**document order** (keyed by claim index, never by completion order). A single
claim's grounding failure is logged and degraded to NOT_ENOUGH_INFO — it never
aborts the whole check.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .config import (
    DEFAULT_N_RUNS,
    MAX_SOURCE_TOKENS,
    SOURCE_CACHE_FLOOR_TOKENS,
    THREAD_POOL_WORKERS,
)
from .decompose import decompose
from .ground import GroundOutcome, ground
from .highlight import highlight_answer
from .llm import LLMProvider, get_provider
from .models import ClaimResult, DecomposedClaim, FaithfulnessReport
from .prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

# Same cheap chars-per-token estimate used by decompose (spec §17): never spend an
# API call just to decide whether the SOURCE is large enough to bother caching.
_CHARS_PER_TOKEN = 4


def check(
    source: str,
    answer: str,
    *,
    n: int = DEFAULT_N_RUNS,
    provider: Optional[LLMProvider] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> FaithfulnessReport:
    """Verify ``answer`` against ``source`` and return a full report.

    Decomposes the answer, grounds each claim ``n`` times (concurrently), scores
    ``SUPPORTED / n_claims`` (``None`` when there are 0 claims — the N/A path), and
    attaches the highlighted answer. ``provider`` defaults to
    :func:`~groundcheck.llm.get_provider`; ``on_progress(done, total)`` is invoked as
    each claim resolves (used by the CLI progress bar and the API/UI).
    """
    provider = provider or get_provider()
    started = time.perf_counter()

    # 0. Cap the SOURCE by token count, mirroring the answer cap in decompose
    #    (spec §17: an oversized source must warn + truncate, never silently balloon
    #    the Opus ×N×claims cost or overflow the context). The answer cap lives in
    #    decompose(); the source enters here, so this is where its cap is enforced.
    source = _truncate_source_if_oversized(source)

    # 1. Decompose (Sonnet). Empty/whitespace answers short-circuit to 0 claims here
    #    with no provider call and cost_usd == 0.0 (decompose owns that).
    dec = decompose(answer, provider=provider)
    claims = dec.decomposition.claims
    total = len(claims)

    # 2. 0 claims → N/A. Same path for an empty/whitespace answer (spec §17, §19-1).
    if total == 0:
        return _assemble(
            claim_results=[],
            answer=answer,
            cost_usd=dec.cost_usd,
            started=started,
            n=n,
        )

    # 3. Ground every claim N times. Results are slotted by claim index so document
    #    order survives out-of-order completion.
    outcomes: list[Optional[GroundOutcome]] = [None] * total
    done = 0

    def record(index: int, outcome: GroundOutcome) -> None:
        nonlocal done
        outcomes[index] = outcome
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    pending = list(range(total))
    if _should_warm_cache_first(source):
        # Ground the first claim synchronously: its first grounding run warms the
        # SOURCE cache so the fanned-out remainder can read it (~0.1× input).
        record(0, _ground_claim_safe(source, claims[0], n, provider))
        pending = list(range(1, total))

    if pending:
        with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as pool:
            futures = {
                pool.submit(_ground_claim_safe, source, claims[i], n, provider): i
                for i in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                record(index, future.result())  # _ground_claim_safe never raises

    # 4. Merge each outcome with its claim, in document order, then assemble.
    claim_results = [
        _to_claim_result(claims[i], outcomes[i])  # type: ignore[arg-type]
        for i in range(total)
    ]
    grounding_cost = sum(o.cost_usd for o in outcomes if o is not None)
    return _assemble(
        claim_results=claim_results,
        answer=answer,
        cost_usd=dec.cost_usd + grounding_cost,
        started=started,
        n=n,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _truncate_source_if_oversized(source: str) -> str:
    """Truncate over-cap sources to ``MAX_SOURCE_TOKENS`` with a warning (spec §17).

    Mirrors ``decompose._truncate_if_oversized`` (the answer cap): never silently
    truncate — the warning names the cut. Uses the cheap char heuristic, not an API
    token count. Returns the source unchanged when it is within the cap.
    """
    max_chars = MAX_SOURCE_TOKENS * _CHARS_PER_TOKEN
    if len(source) <= max_chars:
        return source
    logger.warning(
        "source is ~%d tokens, over the %d-token cap; truncating to the cap before "
        "grounding (text past the cut is dropped — shorten the source to avoid this)",
        len(source) // _CHARS_PER_TOKEN,
        MAX_SOURCE_TOKENS,
    )
    return source[:max_chars]


def _should_warm_cache_first(source: str) -> bool:
    """True when the source is big enough to cache (≥ the 4096-token Opus floor).

    Below the floor there is no caching benefit, so the pipeline fans out
    immediately; at/above it, warm one grounding call before fanning out (spec §10).
    """
    est_source_tokens = len(source) // _CHARS_PER_TOKEN
    return est_source_tokens >= SOURCE_CACHE_FLOOR_TOKENS


def _ground_claim_safe(
    source: str, claim: DecomposedClaim, n: int, provider: LLMProvider
) -> GroundOutcome:
    """Ground one claim, degrading any hard failure to NEI (never raises).

    ``ground`` already absorbs refusals and unparseable responses per run (→ NEI);
    this guard only catches a genuine exception (e.g. a network error surviving the
    SDK's retries) so one bad claim cannot abort the whole check (spec §17).
    """
    try:
        return ground(source, claim.claim, n=n, provider=provider)
    except Exception:  # noqa: BLE001 — deliberate: degrade, don't crash the check
        logger.warning(
            "grounding failed for claim %r; degrading to NOT_ENOUGH_INFO",
            claim.claim,
            exc_info=True,
        )
        return GroundOutcome(
            label="NOT_ENOUGH_INFO",
            supporting_span="",
            rationale="grounding failed for this claim; treated as NOT_ENOUGH_INFO",
            votes={"NOT_ENOUGH_INFO": max(n, 1)},
            confidence=0.0,  # honest: we have no real confidence in a failed claim
            refused=False,
            cost_usd=0.0,  # the failed call's usage is lost; don't invent a cost
            n_refused_runs=0,
        )


def _to_claim_result(claim: DecomposedClaim, outcome: GroundOutcome) -> ClaimResult:
    """Merge a grounding outcome with its claim's text/source-sentence."""
    return ClaimResult(
        claim=claim.claim,
        source_sentence=claim.source_sentence,
        label=outcome.label,
        supporting_span=outcome.supporting_span,
        rationale=outcome.rationale,
        votes=outcome.votes,
        confidence=outcome.confidence,
        refused=outcome.refused,
    )


def _assemble(
    *,
    claim_results: list[ClaimResult],
    answer: str,
    cost_usd: float,
    started: float,
    n: int,
) -> FaithfulnessReport:
    """Build the report (counts + score) and attach the highlighted answer."""
    report = FaithfulnessReport.from_claims(
        claim_results,
        cost_usd=cost_usd,
        latency_s=time.perf_counter() - started,
        n_runs=n,
        prompt_version=PROMPT_VERSION,
    )
    report.highlighted_html, report.unlocated_sentences = highlight_answer(
        answer, claim_results
    )
    return report
