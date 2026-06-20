"""Static/contract checks that the dc-html frontend is wired to the live API (Split 09).

Browser rendering of a ``.dc.html`` (custom dc-runtime + CDN React) cannot be unit
tested in this repo's toolchain — full automated browser e2e is Split 11. These checks
read ``app/GroundCheck.dc.html`` as text and assert the wiring is in place:

* ``runCheck`` ``fetch``es ``/check`` with a ``{source, answer, n}`` JSON body;
* the report→component mapper references the engine fields it maps
  (``supporting_span``, ``votes``, ``confidence``, ``faithfulness_score``);
* the hardcoded *result* array and the ``resolveOrder`` reveal-order literal are gone;
* ``this.SRC`` / ``this.ANS`` remain as prefill **and are byte-identical** to
  ``core/examples/example_hallucinated.json`` (the mock keys grounding on the answer
  string, so any drift silently breaks the 62% in mock mode).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HTML = _REPO_ROOT / "app" / "GroundCheck.dc.html"
_EXAMPLE = _REPO_ROOT / "core" / "examples" / "example_hallucinated.json"


@pytest.fixture(scope="module")
def html() -> str:
    return _HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(html: str) -> str:
    """The embedded ``<script type="text/x-dc">`` block (the only file we edit)."""
    m = re.search(
        r'<script type="text/x-dc"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    assert m, "could not find the <script type=\"text/x-dc\"> block"
    return m.group(1)


# --------------------------------------------------------------------------- #
# 1. runCheck fetches POST /check with the {source, answer, n} body
# --------------------------------------------------------------------------- #
def test_fetches_check_endpoint(script: str) -> None:
    assert "fetch(" in script, "runCheck must call fetch()"
    assert "/check" in script, "the fetch path must include /check"


def test_request_body_has_source_answer_n(script: str) -> None:
    m = re.search(r"JSON\.stringify\(\s*\{(.*?)\}\s*\)", script, re.DOTALL)
    assert m, "the POST body must be built with JSON.stringify({...})"
    body = m.group(1)
    for key in ("source", "answer", "n"):
        assert re.search(rf"\b{key}\s*:", body), f"request body missing '{key}'"


def test_posts_with_json_content_type(script: str) -> None:
    assert '"Content-Type"' in script and "application/json" in script


# --------------------------------------------------------------------------- #
# 2. the report->component mapping references the engine fields it maps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field", ["supporting_span", "votes", "confidence", "faithfulness_score"]
)
def test_mapping_references_report_fields(script: str, field: str) -> None:
    assert field in script, f"the mapping must reference report field '{field}'"


def test_uses_math_floor_not_round_for_score(script: str) -> None:
    # 5/8 = 0.625 must print 62%, not 63% (Math.round(62.5) === 63).
    assert "Math.floor" in script
    # Strip // line comments first (a comment may legitimately *name* Math.round
    # while the code never calls it) — there are no `//` inside string literals here.
    code = re.sub(r"//.*", "", script)
    assert "Math.round" not in code, "score must use Math.floor, never Math.round"


# --------------------------------------------------------------------------- #
# 3. the hardcoded result array + reveal-order literal are gone
# --------------------------------------------------------------------------- #
def test_hardcoded_result_array_removed(script: str) -> None:
    # The old source-of-truth was `this.claims = [ { ... label: ... }, ... ]`.
    assert re.search(r"this\.claims\s*=\s*\[\s*\{", script) is None, (
        "the hardcoded result `this.claims = [{...}]` array must be removed"
    )
    # And no result-object literals should survive in the constructor.
    assert 'label:"SUPPORTED"' not in script
    assert 'label:"NOT_ENOUGH_INFO"' not in script


def test_resolve_order_literal_removed(script: str) -> None:
    compact = re.sub(r"\s+", "", script)
    assert "resolveOrder=[0,1,2,3,6,7,4,5]" not in compact, (
        "the hardcoded `this.resolveOrder = [0,1,2,3,6,7,4,5]` literal must be removed"
    )
    # It is now derived from the live claim count.
    assert "mapped.map" in script, "resolveOrder should be derived from mapped claims"


# --------------------------------------------------------------------------- #
# 4. SRC/ANS remain as prefill and match the canonical example byte-for-byte
# --------------------------------------------------------------------------- #
def test_src_ans_retained(script: str) -> None:
    assert "this.SRC" in script and "this.ANS" in script


def _extract(script: str, name: str) -> str:
    m = re.search(rf'this\.{name} = "(.*?)";', script)
    assert m, f"could not extract this.{name}"
    return m.group(1)


def test_src_ans_byte_identical_to_example(script: str) -> None:
    example = json.loads(_EXAMPLE.read_text(encoding="utf-8"))
    assert _extract(script, "SRC") == example["source"], (
        "this.SRC drifted from example_hallucinated.json's source — "
        "mock mode keys on the answer string, so this would break the 62%"
    )
    assert _extract(script, "ANS") == example["answer"], (
        "this.ANS drifted from example_hallucinated.json's answer"
    )


# --------------------------------------------------------------------------- #
# 5. error capture exists so a failed/again-clicked check never hangs
# --------------------------------------------------------------------------- #
def test_error_state_wired(script: str) -> None:
    # On failure the component must enter phase "error" (never stay "checking").
    assert 'phase:"error"' in script
    # And the markup must surface it with a recoverable retry path.
    assert "isError" in script


def test_error_block_in_markup(html: str) -> None:
    assert 'value="{{ isError }}"' in html, "an isError display block must exist"
    assert "{{ errorMsg }}" in html, "the error message must be bound in the markup"


# --------------------------------------------------------------------------- #
# 6. Split 10 — every live edge state is wired, and the two shipped-mockup a11y
#    bugs (palette word, scoreDisplay) are fixed. (Behaviour is verified against
#    the live mock API in tmp/verify_split10.mjs; these pin the wiring statically.)
# --------------------------------------------------------------------------- #
def test_na_branch_exists(script: str) -> None:
    # Score is driven by faithfulness_score and short-circuits to "N/A" on null.
    assert "faithfulness_score" in script
    assert '"N/A"' in script, "an N/A score branch must exist"
    assert "isNA" in script, "an explicit isNA flag must drive the N/A treatment"


def test_no_unguarded_percentage_division(script: str) -> None:
    # Every percentage uses an `nClaims ?` guard so an empty answer never yields NaN%.
    assert re.search(r"nClaims\s*\?", script), "percentage math must guard nClaims > 0"


@pytest.mark.parametrize(
    "token", ["n_refused", "unlocated_sentences", "warnings", "missing_api_key"]
)
def test_edge_state_fields_referenced(script: str, token: str) -> None:
    assert token in script, f"the script must reference '{token}' to wire its state"


def test_palette_entries_each_define_word(script: str) -> None:
    # Bug 1: every palette() verdict entry must define a `word:` so v.word is never
    # `undefined` — the color-INDEPENDENT verdict (a word + a shape, UI spec §9).
    # Six entries exist (three verdicts × light/dark); all must carry word:.
    entries = re.findall(
        r"\b(SUPPORTED|NOT_ENOUGH_INFO|CONTRADICTED)\s*:\s*\{([^}]*)\}", script
    )
    assert len(entries) == 6, f"expected 6 palette object literals, found {len(entries)}"
    missing = [label for label, obj in entries if "word:" not in obj]
    assert not missing, f"palette entries missing a word: key: {missing}"


def test_score_display_assigned_from_real_score(script: str) -> None:
    # Bug 2: scoreDisplay must be set from the real score (finalScore), not the
    # initial 0 — so the aria-live announces "Faithfulness 62 percent", not 0.
    assert re.search(r"scoreDisplay\s*:\s*finalScore", script), (
        "scoreDisplay must be assigned from the real (finalScore) value"
    )
    assert re.search(r"finalScore\s*=", script), "finalScore must be derived from the score"


def test_refused_treatment_is_monochrome(script: str) -> None:
    # Refused → "◻ declined" (a glyph + a word in ink-3) + the dotted treatment;
    # no 4th/5th color is introduced (principle 1).
    assert "declined" in script
    assert "◻" in script, "the hollow-square glyph (a shape, not a hue) must be present"
    assert "c.refused" in script


def test_live_states_have_markup_regions(html: str) -> None:
    # The new state regions exist in the <x-dc> template (minimal added markup).
    for binding in (
        "{{ naClaims }}",          # N/A claims calm line
        "{{ subText }}",           # N/A subline + refusal attribution
        "{{ hasUnlocated }}",      # unlocated footnote
        "{{ unlocatedNote }}",
        "{{ hasWarnings }}",       # oversize warning
        "{{ warningNote }}",
        "{{ hasMeter }}",          # meter gated off on N/A
        "{{ showRefusalStar }}",   # the "*" on the % for refusal-affected
        "{{ c.wordColor }}",       # per-row word color (ink-3 for declined)
    ):
        assert binding in html, f"missing live-state binding in markup: {binding}"
    assert "No checkable claims were extracted from this answer." in html


def test_na_claims_calm_line_copy(html: str) -> None:
    assert "No checkable claims were extracted from this answer." in html


def test_focus_rings_on_result_controls(html: str) -> None:
    # The two textareas already had style-focus; Split 10 adds an explicit 2px ink
    # ring to the result-area controls (sentence marks + claim rows) — at least 4.
    assert html.count("style-focus=") >= 4, "result-area controls need a visible focus ring"


def test_template_tags_balanced(html: str) -> None:
    # Guard the minimal markup additions: the template region's control tags balance.
    tpl = html.split('<script type="text/x-dc"')[0]
    for tag in ("sc-if", "sc-for", "section", "footer", "div"):
        opens = len(re.findall(rf"<{tag}[ >]", tpl))
        closes = len(re.findall(rf"</{tag}>", tpl))
        assert opens == closes, f"<{tag}> tags unbalanced: {opens} open vs {closes} close"
