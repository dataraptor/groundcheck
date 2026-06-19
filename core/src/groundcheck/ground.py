"""Step 2 — ground one claim against the source, N times, and resolve a verdict.

The grounding judge (Opus 4.8, structured output) is non-deterministic — Opus 4.8
rejects ``temperature``/``top_p``/``top_k`` and has no ``seed`` (README API facts).
Determinism is *engineered* (spec §9): each claim is grounded **N times** and a
majority label is taken, with a **severity tie-break** that biases a split vote
toward *flagging* (CONTRADICTED > NOT_ENOUGH_INFO > SUPPORTED) — the correct bias
for a firewall, and the reason we cannot use ``Counter.most_common`` (spec §19-2,
which is arbitrary on ties).

This module owns the per-claim grounding logic only. It runs the N calls
**sequentially**; the concurrent fan-out (and the warm-one-call-first caching gate)
lives in the pipeline (Split 05), which also merges the returned :class:`GroundOutcome`
with the claim's ``source_sentence`` to build a ``ClaimResult``.

Refusals never crash: Opus can return ``stop_reason == "refusal"`` on a benign
medical claim, so a refused run is mapped to NOT_ENOUGH_INFO + ``refused=True`` and
the rationale ``"model declined to judge"`` (spec §17), checked *before* any parsed
content is read. An empty source is valid input — every claim simply grounds to NEI;
that is the model's job, not a special case here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_N_RUNS, GROUND_MAX_TOKENS, GROUND_MODEL, SEVERITY_ORDER
from .llm import LLMProvider, Usage, compute_cost, get_provider
from .models import GroundingVerdict
from .prompts import GROUND_SYSTEM, ground_claim_block, ground_source_block

# Exact rationale recorded when the judge declines (spec §17 — pinned wording).
REFUSAL_RATIONALE = "model declined to judge"


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O — the heart of the determinism logic)
# --------------------------------------------------------------------------- #


def majority_label(labels: list[str]) -> str:
    """Most-common label; on a tie for the max count, the MOST SEVERE wins (spec §9).

    Counts the labels, finds the max count, and among the labels tied at that count
    returns the one highest in :data:`~groundcheck.config.SEVERITY_ORDER`
    (CONTRADICTED > NOT_ENOUGH_INFO > SUPPORTED). This deliberately replaces
    ``Counter.most_common(1)``, which breaks ties arbitrarily (spec §19-2): a split
    vote must bias toward *flagging*, never silently certify SUPPORTED. A 1-1-1 vote
    resolves to the most severe label **present** (CONTRADICTED only if it got a vote).
    """
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    return max(tied, key=lambda label: SEVERITY_ORDER[label])


def confidence(votes: dict[str, int], n_runs: int) -> float:
    """Per-claim confidence = ``max_votes / n_runs`` (3-0 → 1.0; 2-1 → 0.67).

    Returns 0.0 for a degenerate ``n_runs <= 0`` rather than dividing by zero;
    ``ground`` always calls it with ``n_runs >= 1`` and a non-empty ``votes``.
    """
    if n_runs <= 0 or not votes:
        return 0.0
    return max(votes.values()) / n_runs


# --------------------------------------------------------------------------- #
# Single grounding call
# --------------------------------------------------------------------------- #


def ground_once(
    source: str, claim: str, *, provider: LLMProvider
) -> tuple[GroundingVerdict, bool, Usage]:
    """One Opus grounding call → ``(verdict, refused, usage)``.

    The SOURCE is sent as a ``cache_control: ephemeral`` user block (so the
    second-and-later calls within a check can read it from cache once it is large
    enough to cache — floor 4096 tokens, README API facts); the CLAIM trails it in a
    separate plain block. A refusal is mapped to NEI **before** any parsed content is
    read (spec §17); a non-refusal that still returns no parseable verdict degrades
    conservatively to NEI rather than crashing.
    """
    user_blocks = [
        {
            "type": "text",
            "text": ground_source_block(source),
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": ground_claim_block(claim)},
    ]
    result = provider.parse(
        model=GROUND_MODEL,
        system=GROUND_SYSTEM,
        user_blocks=user_blocks,
        output_model=GroundingVerdict,
        max_tokens=GROUND_MAX_TOKENS,
    )
    if result.refused:
        verdict = GroundingVerdict(
            label="NOT_ENOUGH_INFO", supporting_span="", rationale=REFUSAL_RATIONALE
        )
        return verdict, True, result.usage

    if isinstance(result.parsed, GroundingVerdict):
        return result.parsed, False, result.usage

    # Non-refusal but unparseable (e.g. truncated): degrade to the conservative
    # default rather than crash. Not flagged as a refusal — it wasn't one.
    verdict = GroundingVerdict(
        label="NOT_ENOUGH_INFO",
        supporting_span="",
        rationale="model returned no parseable verdict",
    )
    return verdict, False, result.usage


# --------------------------------------------------------------------------- #
# N-run aggregator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroundOutcome:
    """The resolved grounding of one claim across N runs (pre-``ClaimResult``).

    The pipeline (Split 05) merges this with the claim's ``claim``/``source_sentence``
    to assemble the public :class:`~groundcheck.models.ClaimResult`.
    """

    label: str
    supporting_span: str  # from a winning-label run; "" when the winner is NEI
    rationale: str
    votes: dict[str, int]  # counts per label across the N runs (present labels only)
    confidence: float
    refused: bool  # any run refused
    cost_usd: float  # summed compute_cost over the N runs
    n_refused_runs: int


def ground(
    source: str,
    claim: str,
    *,
    n: int = DEFAULT_N_RUNS,
    provider: LLMProvider | None = None,
) -> GroundOutcome:
    """Ground ``claim`` against ``source`` N times and resolve the majority verdict.

    Runs :func:`ground_once` ``n`` times **sequentially** (the pipeline owns the
    concurrent fan-out, Split 05), takes the :func:`majority_label` with the severity
    tie-break, computes :func:`confidence` from the vote spread, surfaces the span /
    rationale from the **first run whose label is the winner** (so a NEI claim never
    shows a SUPPORTED span — and its span is forced to ``""``), and sums the per-run
    cost. Never raises on an empty source or a refusal; it degrades.
    """
    provider = provider or get_provider()

    verdicts: list[GroundingVerdict] = []
    refused_flags: list[bool] = []
    cost_usd = 0.0
    for _ in range(n):
        verdict, refused, usage = ground_once(source, claim, provider=provider)
        verdicts.append(verdict)
        refused_flags.append(refused)
        cost_usd += compute_cost(GROUND_MODEL, usage)

    labels = [v.label for v in verdicts]
    votes: dict[str, int] = {}
    for label in labels:
        votes[label] = votes.get(label, 0) + 1

    winner = majority_label(labels)
    representative = next(v for v in verdicts if v.label == winner)
    span = "" if winner == "NOT_ENOUGH_INFO" else representative.supporting_span

    return GroundOutcome(
        label=winner,
        supporting_span=span,
        rationale=representative.rationale,
        votes=votes,
        confidence=confidence(votes, n),
        refused=any(refused_flags),
        cost_usd=cost_usd,
        n_refused_runs=sum(refused_flags),
    )
