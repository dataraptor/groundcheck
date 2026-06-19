"""Canonical prompts, pinned verbatim from spec §6.

`PROMPT_VERSION` is recorded in every report and eval run for reproducibility.
The two system prompts are byte-identical to §6 — do not paraphrase. The user-content
builders pin the SOURCE/CLAIM/ANSWER wording so it isn't re-invented mid-build; the
actual `cache_control` placement on the SOURCE block lands in Split 04.
"""

from __future__ import annotations

PROMPT_VERSION = "v3"

# Decompose (Sonnet 4.6) — system. Verbatim from spec §6.
DECOMPOSE_SYSTEM = """\
You break an AI-generated ANSWER into atomic factual claims so each can be
checked against a source document.

Output ONLY atomic, independently checkable factual assertions:
- One fact per claim. Split compound sentences ("X and Y") into separate claims.
- Keep a single quantity, ratio, or measurement intact ("130/80 mm Hg", "by 25%")
  — never split a number from its unit or its subject.
- IGNORE anything that is not a checkable factual assertion: questions, hedges,
  opinions, advice/recommendations to the reader, pleasantries, meta-commentary.
- Do NOT add facts not present in the answer. Do NOT correct or fact-check here —
  only extract.

For each claim, copy the exact sentence from the ANSWER it came from into
source_sentence (verbatim, including punctuation) so the claim can be traced
back for highlighting.

If the answer contains no checkable factual claims, return an empty list."""

# Ground (Opus 4.8) — system. Verbatim from spec §6.
GROUND_SYSTEM = """\
You are a strict grounding verifier. You are given a SOURCE document and a single
CLAIM. Decide whether the SOURCE supports the claim.

Judge ONLY against the SOURCE. Never use outside or world knowledge. A claim can
be true in reality and still be NOT_ENOUGH_INFO when the source does not establish
it — that is the correct label.

Labels:
- SUPPORTED: the source explicitly states the claim, or it follows directly and
  unambiguously from the source.
- CONTRADICTED: the source asserts something that conflicts with the claim
  (a different number, the opposite direction, a mutually exclusive fact).
- NOT_ENOUGH_INFO: the source neither establishes nor conflicts with the claim.
  This is the default whenever the source is silent or only partially relevant.

Be conservative: if the source does not clearly establish the claim, use
NOT_ENOUGH_INFO rather than guessing SUPPORTED.

supporting_span: copy the exact span from the SOURCE that supports or contradicts
the claim; empty string for NOT_ENOUGH_INFO.
rationale: one sentence, grounded in the source."""


def decompose_user(answer: str) -> str:
    """User content for the decompose call: ``ANSWER:\\n<answer>`` (spec §6)."""
    return f"ANSWER:\n{answer}"


def ground_source_block(source: str) -> str:
    """The cached SOURCE prefix of the ground user content: ``SOURCE:\\n<source>`` (spec §6/§10)."""
    return f"SOURCE:\n{source}"


def ground_claim_block(claim: str) -> str:
    """The trailing CLAIM block of the ground user content: ``\\n\\nCLAIM:\\n<claim>`` (spec §6)."""
    return f"\n\nCLAIM:\n{claim}"
