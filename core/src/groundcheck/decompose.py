"""Step 1 — decompose an answer into atomic claims (spec §5/§6/§17).

``decompose(answer)`` runs the Sonnet 4.6 structured-output call through the
provider seam and returns a :class:`DecomposeOutcome` that carries *both* the
parsed :class:`~groundcheck.models.Decomposition` and the call's USD cost — so the
pipeline (Split 05) can fold the decompose cost into the per-check total instead of
silently under-counting it (honest accounting, spec §10; mirrors ``ground()`` →
``GroundOutcome``).

Edge cases never crash (spec §17):

* empty / whitespace answer → 0 claims, ``cost_usd == 0.0``, **no provider call**;
* oversized answer → truncate to the cap with a warning, then proceed;
* model truncation (``max_tokens``) / refusal → warn and degrade to best-effort.

No network import lives at module top level — the provider lazy-imports its SDK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import DECOMPOSE_MAX_TOKENS, DECOMPOSE_MODEL, MAX_ANSWER_TOKENS
from .llm import LLMProvider, compute_cost, get_provider
from .models import Decomposition
from .prompts import DECOMPOSE_SYSTEM, decompose_user

logger = logging.getLogger(__name__)

# Cheap chars-per-token estimate (spec §17: estimate the size, never spend an API
# call counting tokens just to decide whether to truncate).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class DecomposeOutcome:
    """The result of one decompose step: the claims plus the call's cost/flags.

    ``cost_usd`` is the USD for the single Sonnet call (``0.0`` on the no-call empty
    path; the real cost otherwise, even when truncated/refused — the call still cost
    money). ``decomposition`` is always a valid :class:`Decomposition`, never ``None``.
    """

    decomposition: Decomposition
    cost_usd: float
    truncated: bool = False  # stop_reason == "max_tokens"
    refused: bool = False  # stop_reason == "refusal" (rare for decompose)


def decompose(
    answer: str,
    *,
    provider: LLMProvider | None = None,
    max_tokens: int = DECOMPOSE_MAX_TOKENS,
) -> DecomposeOutcome:
    """Answer → atomic claims (Sonnet 4.6, structured output).

    See the module docstring for the edge-case contract. Pass ``provider`` to
    inject a stub/mock; it defaults to :func:`~groundcheck.llm.get_provider`.
    """
    # Empty / whitespace answer → 0 claims → N/A, WITHOUT spending a call (spec §17).
    # Checked before resolving the provider so an injected provider is never touched.
    if not answer or not answer.strip():
        return DecomposeOutcome(Decomposition(claims=[]), cost_usd=0.0)

    provider = provider or get_provider()
    answer = _truncate_if_oversized(answer)

    result = provider.parse(
        model=DECOMPOSE_MODEL,
        system=DECOMPOSE_SYSTEM,
        user_blocks=[{"type": "text", "text": decompose_user(answer)}],
        output_model=Decomposition,
        max_tokens=max_tokens,
    )
    # The call cost is carried on every branch below (even truncated/refused).
    cost = compute_cost(DECOMPOSE_MODEL, result.usage)

    if result.refused:
        logger.warning("decompose was refused by the model; returning 0 claims (best-effort).")
        return DecomposeOutcome(Decomposition(claims=[]), cost_usd=cost, refused=True)

    parsed = result.parsed if isinstance(result.parsed, Decomposition) else Decomposition(claims=[])

    if result.truncated:
        logger.warning(
            "decompose hit max_tokens; claims may be incomplete — consider raising "
            "max_tokens or shortening the answer"
        )
        return DecomposeOutcome(parsed, cost_usd=cost, truncated=True)

    return DecomposeOutcome(parsed, cost_usd=cost)


def _truncate_if_oversized(answer: str) -> str:
    """Truncate over-cap answers to ``MAX_ANSWER_TOKENS`` with a warning (spec §17).

    Never silently truncate — the warning names the cut. Uses the cheap char
    heuristic, not an API token count.
    """
    max_chars = MAX_ANSWER_TOKENS * _CHARS_PER_TOKEN
    if len(answer) <= max_chars:
        return answer
    logger.warning(
        "answer is ~%d tokens, over the %d-token cap; truncating to the cap before "
        "decompose (claims past the cut are dropped — shorten the answer to avoid this)",
        len(answer) // _CHARS_PER_TOKEN,
        MAX_ANSWER_TOKENS,
    )
    return answer[:max_chars]
