"""Tests for Split 05 — the whitespace-tolerant highlighter (spec §8). No key needed."""

from __future__ import annotations

from groundcheck.config import VERDICT_COLORS
from groundcheck.highlight import highlight_answer
from groundcheck.models import ClaimResult

GREEN = VERDICT_COLORS["SUPPORTED"]
AMBER = VERDICT_COLORS["NOT_ENOUGH_INFO"]
RED = VERDICT_COLORS["CONTRADICTED"]


def _claim(label: str, source_sentence: str, *, claim: str = "c") -> ClaimResult:
    return ClaimResult(
        claim=claim,
        source_sentence=source_sentence,
        label=label,
        supporting_span="" if label == "NOT_ENOUGH_INFO" else "span",
        rationale="r",
        votes={label: 3},
        confidence=1.0,
    )


# --------------------------------------------------------------------------- #
# Rung (a): exact match
# --------------------------------------------------------------------------- #


def test_exact_match():
    answer = "The sky is blue. Grass is green."
    claims = [_claim("SUPPORTED", "The sky is blue.")]
    html, unlocated = highlight_answer(answer, claims)
    assert unlocated == []
    assert f'background:{GREEN}' in html
    assert "The sky is blue." in html  # the located text is present, wrapped
    # The unhighlighted remainder is still emitted verbatim (escaped).
    assert "Grass is green." in html


# --------------------------------------------------------------------------- #
# Rung (b): whitespace-tolerant regex (newline / collapsed spaces)
# --------------------------------------------------------------------------- #


def test_whitespace_tolerant_match():
    # The answer has a newline + double space inside the sentence; the stored
    # source_sentence uses single spaces. Exact find fails; the regex path locates it.
    answer = "A reading of 130/80 mm Hg\nor  higher is considered high."
    stored = "A reading of 130/80 mm Hg or higher is considered high."
    html, unlocated = highlight_answer(answer, [_claim("SUPPORTED", stored)])
    assert unlocated == []
    assert f'background:{GREEN}' in html
    # Original offsets preserved → the answer's own whitespace survives inside the span.
    assert "130/80 mm Hg" in html


def test_prefix_fallback_locates_partial():
    # Exact and full-sentence regex both fail (the tail diverges), but the first ~40
    # non-space chars share a clean prefix → rung (c) anchors it. The shared prefix
    # ("...symptoms whatsoever", ~47 non-space chars) exceeds the 40-char window.
    stored = "High blood pressure usually has no symptoms whatsoever in adult patients."
    answer = "High blood pressure usually has no symptoms whatsoever, the doctor explained."
    html, unlocated = highlight_answer(answer, [_claim("SUPPORTED", stored)])
    assert unlocated == []
    assert f'background:{GREEN}' in html


# --------------------------------------------------------------------------- #
# Duplicates: first occurrence only
# --------------------------------------------------------------------------- #


def test_duplicate_first_occurrence():
    answer = "Repeat me. Then other text. Repeat me."
    html, unlocated = highlight_answer(answer, [_claim("SUPPORTED", "Repeat me.")])
    assert unlocated == []
    # Exactly one wrap (the first occurrence); the second stays plain text.
    assert html.count(f'background:{GREEN}') == 1
    # The span opens at the very start (first occurrence).
    assert html.startswith(f'<span style="background:{GREEN}')


# --------------------------------------------------------------------------- #
# Overlaps: second overlapping span skipped
# --------------------------------------------------------------------------- #


def test_overlap_skipped():
    answer = "the quick brown fox jumps over"
    # Two source_sentences whose matches overlap on "brown fox".
    claims = [
        _claim("SUPPORTED", "the quick brown fox", claim="a"),
        _claim("NOT_ENOUGH_INFO", "brown fox jumps over", claim="b"),
    ]
    html, unlocated = highlight_answer(answer, claims)
    assert unlocated == []
    # First (by offset) is emitted; the overlapping second is skipped → one wrap only,
    # and no amber wrap (the NEI sentence overlapped and was dropped).
    assert html.count("<span") == 1
    assert f'background:{GREEN}' in html
    assert f'background:{AMBER}' not in html


# --------------------------------------------------------------------------- #
# No match → unlocated, unhighlighted, no crash
# --------------------------------------------------------------------------- #


def test_no_match_goes_unlocated():
    answer = "Completely different answer text."
    missing = "This sentence is absent from the answer."
    html, unlocated = highlight_answer(answer, [_claim("SUPPORTED", missing)])
    assert unlocated == [missing]
    assert "<span" not in html  # nothing highlighted
    assert html == "Completely different answer text."  # escaped passthrough (no specials)


# --------------------------------------------------------------------------- #
# HTML escaping
# --------------------------------------------------------------------------- #


def test_html_escaping():
    answer = "Tom & Jerry <b>win</b> 3 > 2 always."
    # Highlight a sentence that itself contains specials.
    html, unlocated = highlight_answer(answer, [_claim("SUPPORTED", "Tom & Jerry <b>win</b>")])
    assert unlocated == []
    # No raw injection anywhere in the output.
    assert "<b>" not in html
    assert "&amp;" in html and "&lt;" in html and "&gt;" in html
    # The colored wrapper span itself is real markup (not escaped).
    assert f'<span style="background:{GREEN}' in html


def test_html_escaping_without_any_match():
    answer = "a < b && c > d"
    html, unlocated = highlight_answer(answer, [])
    assert unlocated == []
    assert html == "a &lt; b &amp;&amp; c &gt; d"


# --------------------------------------------------------------------------- #
# Worst-verdict-per-sentence coloring
# --------------------------------------------------------------------------- #


def test_worst_verdict_color_supported_plus_nei_is_amber():
    answer = "Shared sentence here."
    claims = [
        _claim("SUPPORTED", "Shared sentence here.", claim="a"),
        _claim("NOT_ENOUGH_INFO", "Shared sentence here.", claim="b"),
    ]
    html, _ = highlight_answer(answer, claims)
    assert f'background:{AMBER}' in html  # NEI is worse than SUPPORTED
    assert f'background:{GREEN}' not in html


def test_worst_verdict_color_supported_plus_contradicted_is_red():
    answer = "Shared sentence here."
    claims = [
        _claim("SUPPORTED", "Shared sentence here.", claim="a"),
        _claim("CONTRADICTED", "Shared sentence here.", claim="b"),
    ]
    html, _ = highlight_answer(answer, claims)
    assert f'background:{RED}' in html  # CONTRADICTED is the most severe
    assert f'background:{GREEN}' not in html
    assert f'background:{AMBER}' not in html


# --------------------------------------------------------------------------- #
# Never raises; empty inputs
# --------------------------------------------------------------------------- #


def test_empty_answer_and_no_claims():
    assert highlight_answer("", []) == ("", [])


def test_blank_source_sentence_is_ignored():
    # A claim with no source_sentence contributes nothing to highlight (no crash).
    html, unlocated = highlight_answer("some text", [_claim("SUPPORTED", "")])
    assert unlocated == []
    assert "<span" not in html
