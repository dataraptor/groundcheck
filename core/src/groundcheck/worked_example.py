"""The canonical §5 worked example (the money demo), in one reusable place.

Every part of the build that has to reproduce the shipped 62% screenshot keys off
the literals here:

* ``MockProvider`` auto-seeds :data:`WORKED_EXAMPLE_DECOMPOSITION` so the no-key
  ``decompose`` returns the eight §5 claims (Split 03).
* Split 04 adds the grounding verdicts for these eight claims here.
* Split 05 authors ``examples/example_hallucinated.json`` from these strings.

⚠️ **Byte-identical to the frontend.** :data:`WORKED_EXAMPLE_SOURCE` and
:data:`WORKED_EXAMPLE_ANSWER` are copied verbatim from ``app/GroundCheck.dc.html``
(``this.SRC`` / ``this.ANS``). The mock keys grounding on the answer string, so any
drift would make mock mode fall back to default verdicts and silently break the
62%. A test pins them against the dc-html literals so they cannot drift.
"""

from __future__ import annotations

from .models import DecomposedClaim, Decomposition

# --- the demo inputs (verbatim from app/GroundCheck.dc.html this.SRC / this.ANS) --- #
WORKED_EXAMPLE_SOURCE = (
    "High blood pressure, also called hypertension, usually has no symptoms. "
    "A reading of 130/80 mm Hg or higher is considered high. Over time, high "
    "blood pressure raises the risk of heart attack, stroke, and kidney disease. "
    "Lifestyle changes such as reducing salt, exercising regularly, and "
    "maintaining a healthy weight can help lower blood pressure. Some people also "
    "need medicine to keep their blood pressure under control."
)

WORKED_EXAMPLE_ANSWER = (
    "Hypertension usually causes no symptoms. Many people don't know they have "
    "it. A reading of 130/80 mm Hg or higher is considered high blood pressure. "
    "It increases the risk of heart attack, stroke, and kidney disease. Cutting "
    "salt lowers blood pressure by exactly 25% in every patient. Hypertension is "
    "the leading cause of death worldwide. Doctors recommend regular exercise and "
    "maintaining a healthy weight. Some patients need medication."
)

# --- the eight atomic claims (spec §5 table; Split-03 deliverable) ----------------- #
# In this worked example the answer is eight one-fact sentences, so each claim's
# ``source_sentence`` is simply the answer sentence it came from (claim == sentence).
# Expected verdicts (claims 2, 5, 6 are NOT_ENOUGH_INFO) are added by Split 04.
WORKED_EXAMPLE_CLAIMS: list[DecomposedClaim] = [
    DecomposedClaim(
        claim="Hypertension usually causes no symptoms.",
        source_sentence="Hypertension usually causes no symptoms.",
    ),
    DecomposedClaim(
        claim="Many people don't know they have it.",
        source_sentence="Many people don't know they have it.",
    ),
    DecomposedClaim(
        claim="A reading of 130/80 mm Hg or higher is considered high blood pressure.",
        source_sentence="A reading of 130/80 mm Hg or higher is considered high blood pressure.",
    ),
    DecomposedClaim(
        claim="It increases the risk of heart attack, stroke, and kidney disease.",
        source_sentence="It increases the risk of heart attack, stroke, and kidney disease.",
    ),
    DecomposedClaim(
        claim="Cutting salt lowers blood pressure by exactly 25% in every patient.",
        source_sentence="Cutting salt lowers blood pressure by exactly 25% in every patient.",
    ),
    DecomposedClaim(
        claim="Hypertension is the leading cause of death worldwide.",
        source_sentence="Hypertension is the leading cause of death worldwide.",
    ),
    DecomposedClaim(
        claim="Doctors recommend regular exercise and maintaining a healthy weight.",
        source_sentence="Doctors recommend regular exercise and maintaining a healthy weight.",
    ),
    DecomposedClaim(
        claim="Some patients need medication.",
        source_sentence="Some patients need medication.",
    ),
]

WORKED_EXAMPLE_DECOMPOSITION = Decomposition(claims=WORKED_EXAMPLE_CLAIMS)

# Mock lookup key: the full answer string. ``MockProvider.register`` normalizes it
# (lowercase + whitespace-collapsed) and matches it as a substring of the request's
# ``ANSWER:\n<answer>`` user text, so a decompose call on this answer resolves here.
WORKED_EXAMPLE_KEY = WORKED_EXAMPLE_ANSWER
