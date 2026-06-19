"""Step 3a — color the answer by per-sentence worst verdict (spec §8).

This is the project's deliberate **soft link**: rule evaluation (the grounding
verdict) is exact, but *locating* a claim's sentence inside the original answer is
best-effort. Locating degrades to "claim shown, not highlighted" — never to a wrong
verdict, and never to a crash.

``highlight_answer(answer, claims) -> (html, unlocated)``:

1. **Group** claims by ``source_sentence``; a sentence's color = the **worst**
   verdict among its claims (CONTRADICTED > NOT_ENOUGH_INFO > SUPPORTED, via
   :data:`~groundcheck.config.SEVERITY_ORDER`).
2. **Locate** each sentence in the original answer with a four-rung ladder that
   preserves original offsets: exact ``find`` → whitespace-tolerant regex →
   ~40-non-space-char prefix → give up (the sentence joins ``unlocated``).
3. **Duplicates:** the **first** occurrence is highlighted (documented limitation).
4. **Overlaps:** spans are emitted left to right; any span overlapping an
   already-emitted one is skipped.
5. **Escaping:** every character of the answer is HTML-escaped; matched spans are
   wrapped in a colored ``<span>`` (the §8 pastels, :data:`config.VERDICT_COLORS`),
   so the result is safe under ``dangerouslySetInnerHTML`` / ``unsafe_allow_html``.

Never raises: the worst case is "everything unhighlighted + a populated ``unlocated``".
"""

from __future__ import annotations

import html as _html
import re

from .config import SEVERITY_ORDER, VERDICT_COLORS
from .models import ClaimResult

# How many leading non-space characters the prefix fallback (rung c) matches on.
_PREFIX_NONSPACE_CHARS = 40

# Span wrapper template (the §8 pastel background + a small rounded chip).
_SPAN_TEMPLATE = '<span style="background:{color};border-radius:3px;padding:0 1px;">{text}</span>'


def highlight_answer(answer: str, claims: list[ClaimResult]) -> tuple[str, list[str]]:
    """Return ``(html, unlocated_source_sentences)`` for ``answer`` (spec §8).

    See the module docstring for the full algorithm. ``html`` is the answer with the
    located source sentences wrapped in worst-verdict colored spans (everything
    escaped); ``unlocated`` lists the source sentences that could not be located
    (their claims still appear in the report — they are simply not highlighted).
    """
    worst_by_sentence = _worst_verdict_by_sentence(claims)

    # Locate each distinct sentence (document order of first appearance).
    located: list[tuple[int, int, str]] = []  # (start, end, label)
    unlocated: list[str] = []
    for sentence, label in worst_by_sentence.items():
        match = _locate(answer, sentence)
        if match is None:
            unlocated.append(sentence)
        else:
            start, end = match
            located.append((start, end, label))

    spans = _drop_overlaps(located)
    return _build_html(answer, spans), unlocated


# --------------------------------------------------------------------------- #
# 1. Worst verdict per source sentence
# --------------------------------------------------------------------------- #


def _worst_verdict_by_sentence(claims: list[ClaimResult]) -> dict[str, str]:
    """Map each non-empty ``source_sentence`` to its most severe claim label.

    Insertion order follows first appearance in ``claims`` (document order), so the
    later left-to-right offset sort is stable for sentences that tie on start.
    """
    worst: dict[str, str] = {}
    for claim in claims:
        sentence = claim.source_sentence
        if not sentence:
            continue  # nothing to highlight against
        current = worst.get(sentence)
        if current is None or SEVERITY_ORDER[claim.label] > SEVERITY_ORDER[current]:
            worst[sentence] = claim.label
    return worst


# --------------------------------------------------------------------------- #
# 2. Locate a sentence: exact -> regex -> prefix -> give up
# --------------------------------------------------------------------------- #


def _locate(answer: str, sentence: str) -> tuple[int, int] | None:
    """Find ``sentence`` in ``answer``, returning original ``(start, end)`` offsets.

    Tries, in order: exact substring; a whitespace-tolerant regex built from the
    sentence's tokens; the same regex over just the first ~40 non-space characters.
    Returns ``None`` (give up) if all three fail — never raises.
    """
    # (a) exact — the fast path when the model copied the sentence verbatim.
    idx = answer.find(sentence)
    if idx != -1:
        return idx, idx + len(sentence)

    # (b) whitespace-tolerant: re.escape each token, join with \s+ so newlines /
    #     collapsed or extra spaces between words still match — at original offsets.
    full = _whitespace_tolerant_search(answer, sentence)
    if full is not None:
        return full

    # (c) prefix on the first ~40 non-space chars (handles a divergent tail).
    prefix = _leading_nonspace(sentence, _PREFIX_NONSPACE_CHARS)
    if prefix and prefix != sentence:
        pref = _whitespace_tolerant_search(answer, prefix)
        if pref is not None:
            return pref

    # (d) give up — caller records it as unlocated (graceful degradation).
    return None


def _whitespace_tolerant_search(answer: str, sentence: str) -> tuple[int, int] | None:
    """Search ``answer`` for ``sentence`` ignoring inter-token whitespace differences."""
    tokens = sentence.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, answer)
    if match is None:
        return None
    return match.start(), match.end()


def _leading_nonspace(text: str, limit: int) -> str:
    """Return the prefix of ``text`` covering its first ``limit`` non-space chars."""
    count = 0
    chars: list[str] = []
    for ch in text:
        chars.append(ch)
        if not ch.isspace():
            count += 1
            if count >= limit:
                break
    return "".join(chars)


# --------------------------------------------------------------------------- #
# 3/4. Order + drop overlaps
# --------------------------------------------------------------------------- #


def _drop_overlaps(located: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Sort spans left to right and skip any that overlaps an already-kept span."""
    kept: list[tuple[int, int, str]] = []
    emitted_end = -1
    for start, end, label in sorted(located, key=lambda s: s[0]):
        if start < emitted_end:
            continue  # overlaps a span we already emitted — skip (spec §8.4)
        kept.append((start, end, label))
        emitted_end = end
    return kept


# --------------------------------------------------------------------------- #
# 5. Build the escaped, span-wrapped HTML
# --------------------------------------------------------------------------- #


def _build_html(answer: str, spans: list[tuple[int, int, str]]) -> str:
    """HTML-escape the whole answer, wrapping the given spans in colored ``<span>``s."""
    out: list[str] = []
    cursor = 0
    for start, end, label in spans:
        out.append(_html.escape(answer[cursor:start]))  # plain text before the span
        out.append(
            _SPAN_TEMPLATE.format(
                color=VERDICT_COLORS[label],
                text=_html.escape(answer[start:end]),
            )
        )
        cursor = end
    out.append(_html.escape(answer[cursor:]))  # trailing plain text
    return "".join(out)
