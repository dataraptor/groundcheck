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

from .models import DecomposedClaim, Decomposition, GroundingVerdict

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


# --- the eight grounding verdicts (spec §5; Split-04 deliverable) ------------------ #
# Five SUPPORTED + three NOT_ENOUGH_INFO → 5/8 = 62% (the money demo). Verdicts key on
# the *claim* text, which MockProvider matches as a substring of each ground call's
# ``CLAIM:`` block (no claim is a verbatim substring of the source, so only its own
# claim matches). ``supporting_span`` is a verbatim SOURCE substring for SUPPORTED and
# "" for NEI (a test pins both invariants).
#
# Claims 2 and 5 are SPLIT VOTES — expressed as an ORDERED SEQUENCE of three verdicts
# ``[NEI, NEI, SUPPORTED]`` that MockProvider consumes by call index, so three
# identical ``ground_once`` calls return different labels in order → votes 2·1,
# confidence 0.67. Claim 6 is a clean ``NEI`` (single verdict → 3·0, confidence 1.0).
# This reproduces the dc-html mockup's hardcoded data and gives n_low_confidence == 2
# in Split 05. ⚠️ Load-bearing: Splits 09/11 assert the "exactly 25%" claim (claim 5)
# expands to 2·1 / 0.67 — which only holds while claim 5 stays a split vote here.


def _supported(span: str, rationale: str) -> GroundingVerdict:
    return GroundingVerdict(label="SUPPORTED", supporting_span=span, rationale=rationale)


def _nei(rationale: str) -> GroundingVerdict:
    return GroundingVerdict(label="NOT_ENOUGH_INFO", supporting_span="", rationale=rationale)


# The spurious minority SUPPORTED vote in a split: it is never surfaced (the winner is
# NEI, and the representative span/rationale come from the first NEI run), so its
# content only needs to be schema-valid and honestly labelled as the noise vote.
_SPURIOUS_SUPPORTED = _supported("", "(spurious minority run — not the surfaced verdict)")

WORKED_EXAMPLE_VERDICTS: dict[str, GroundingVerdict | list[GroundingVerdict]] = {
    # 1 — SUPPORTED (clean 3·0)
    WORKED_EXAMPLE_CLAIMS[0].claim: _supported(
        "usually has no symptoms",
        "The source states hypertension usually has no symptoms.",
    ),
    # 2 — NOT_ENOUGH_INFO (split vote 2·1 → confidence 0.67)
    WORKED_EXAMPLE_CLAIMS[1].claim: [
        _nei("The source is silent on awareness."),
        _nei("The source is silent on awareness."),
        _SPURIOUS_SUPPORTED,
    ],
    # 3 — SUPPORTED (clean 3·0)
    WORKED_EXAMPLE_CLAIMS[2].claim: _supported(
        "A reading of 130/80 mm Hg or higher is considered high.",
        "Stated verbatim in the source.",
    ),
    # 4 — SUPPORTED (clean 3·0)
    WORKED_EXAMPLE_CLAIMS[3].claim: _supported(
        "raises the risk of heart attack, stroke, and kidney disease",
        "The source lists these risks.",
    ),
    # 5 — NOT_ENOUGH_INFO (split vote 2·1 → confidence 0.67) — the fabricated "exactly 25%"
    WORKED_EXAMPLE_CLAIMS[4].claim: [
        _nei("The source gives no magnitude for reducing salt."),
        _nei("The source gives no magnitude for reducing salt."),
        _SPURIOUS_SUPPORTED,
    ],
    # 6 — NOT_ENOUGH_INFO (clean 3·0) — the fabricated "leading cause of death"
    WORKED_EXAMPLE_CLAIMS[5].claim: _nei("The source is silent on mortality ranking."),
    # 7 — SUPPORTED (clean 3·0)
    WORKED_EXAMPLE_CLAIMS[6].claim: _supported(
        "Lifestyle changes such as reducing salt, exercising regularly, and "
        "maintaining a healthy weight can help lower blood pressure.",
        "The source recommends exercise and a healthy weight.",
    ),
    # 8 — SUPPORTED (clean 3·0)
    WORKED_EXAMPLE_CLAIMS[7].claim: _supported(
        "Some people also need medicine to keep their blood pressure under control.",
        "The source notes some people also need medicine.",
    ),
}

# Pinned, documented trigger for the refusal → NEI path (spec §17). Grounding any claim
# whose text contains this phrase makes every run refuse, so the GroundOutcome is NEI
# with refused=True. Split 10 drives the "refusal-affected" UI state by feeding an
# answer that decomposes into a claim carrying this phrase — no key-guessing needed.
REFUSAL_TRIGGER = "please refuse to judge this claim"


# --------------------------------------------------------------------------- #
# The fully-grounded counter-example (Split 05 — examples/example_grounded.json)
# --------------------------------------------------------------------------- #
# A second answer over the *same* hypertension source whose every claim is genuinely
# SUPPORTED, so it scores 100% — the "clean" demo beside the hallucinated one. Per
# spec B.2's authoring caution, every sentence paraphrases ONLY a fact the source
# states, and every supporting span below is a verbatim SOURCE substring (a test
# pins both). The answer is written as already-atomic one-fact sentences so the mock
# decomposition is one claim per sentence (claim == source_sentence), mirroring the
# hallucinated fixture. ⚠️ Mock matching is substring-based on the *claim* text, so
# these claims are worded to share no claim/answer substring with the hallucinated
# set (a collision would let the wrong canned verdict answer a grounding call).

GROUNDED_EXAMPLE_SOURCE = WORKED_EXAMPLE_SOURCE  # same source document

GROUNDED_EXAMPLE_ANSWER = (
    "Hypertension is another name for high blood pressure. High blood pressure "
    "usually has no symptoms. A blood pressure reading of 130/80 mm Hg or higher is "
    "high. High blood pressure can raise the risk of heart attack. It can also raise "
    "the risk of stroke. It can raise the risk of kidney disease. Reducing salt can "
    "help lower blood pressure. Exercising regularly can help lower blood pressure. "
    "Keeping a healthy weight can help lower blood pressure. Some people need "
    "medicine to control their blood pressure."
)


def _grounded_claim(text: str) -> DecomposedClaim:
    # Each grounded answer sentence is itself one atomic claim.
    return DecomposedClaim(claim=text, source_sentence=text)


# (claim text, verbatim SOURCE span supporting it) — all SUPPORTED.
_GROUNDED_PAIRS: list[tuple[str, str]] = [
    (
        "Hypertension is another name for high blood pressure.",
        "High blood pressure, also called hypertension",
    ),
    ("High blood pressure usually has no symptoms.", "usually has no symptoms"),
    (
        "A blood pressure reading of 130/80 mm Hg or higher is high.",
        "A reading of 130/80 mm Hg or higher is considered high.",
    ),
    (
        "High blood pressure can raise the risk of heart attack.",
        "raises the risk of heart attack, stroke, and kidney disease",
    ),
    (
        "It can also raise the risk of stroke.",
        "raises the risk of heart attack, stroke, and kidney disease",
    ),
    (
        "It can raise the risk of kidney disease.",
        "raises the risk of heart attack, stroke, and kidney disease",
    ),
    (
        "Reducing salt can help lower blood pressure.",
        "reducing salt, exercising regularly, and maintaining a healthy weight can "
        "help lower blood pressure",
    ),
    (
        "Exercising regularly can help lower blood pressure.",
        "reducing salt, exercising regularly, and maintaining a healthy weight can "
        "help lower blood pressure",
    ),
    (
        # "Keeping" (not "Maintaining") so this claim is NOT a verbatim substring of
        # the SOURCE — the mock matches keys against SOURCE+CLAIM, so a source-substring
        # key would hijack every grounding call (test_no_verdict_key_is_source_substring).
        "Keeping a healthy weight can help lower blood pressure.",
        "reducing salt, exercising regularly, and maintaining a healthy weight can "
        "help lower blood pressure",
    ),
    (
        "Some people need medicine to control their blood pressure.",
        "Some people also need medicine to keep their blood pressure under control.",
    ),
]

GROUNDED_EXAMPLE_CLAIMS: list[DecomposedClaim] = [
    _grounded_claim(text) for text, _span in _GROUNDED_PAIRS
]

GROUNDED_EXAMPLE_DECOMPOSITION = Decomposition(claims=GROUNDED_EXAMPLE_CLAIMS)

GROUNDED_EXAMPLE_KEY = GROUNDED_EXAMPLE_ANSWER

GROUNDED_EXAMPLE_VERDICTS: dict[str, GroundingVerdict] = {
    text: _supported(span, "Stated in the source.") for text, span in _GROUNDED_PAIRS
}
