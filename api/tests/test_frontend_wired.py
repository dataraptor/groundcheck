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
