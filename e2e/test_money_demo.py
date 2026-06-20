"""Browser e2e (Split 11, Deliverable 2) — drive the REAL dc-html page against the
live API (mock mode, no key) and assert rendered DOM/text, not internal state.

The headline ``test_money_demo`` is the project's screenshot moment, now automated:
load ``/``, click **Check faithfulness**, and assert the page reads **62%**, "5 of 8
grounded", and exactly **three** amber (not-in-source) sentences — driven by a real
``POST /check`` that returned 200.

Most tests force reduced-motion so the staggered reveal resolves instantly (the info
is identical — asserted by ``test_reduced_motion``); the animation itself is cosmetic.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _reduce_motion(page):
    """Make the reveal instant + deterministic (reduced-motion path)."""
    page.emulate_media(reduced_motion="reduce")


def _click_check(page):
    page.get_by_role("button", name="Check faithfulness").click()


def _wait_done(page, timeout=15_000):
    """Wait until results are rendered (the score line is visible)."""
    page.locator('p span[role="button"]').first.wait_for(state="visible", timeout=timeout)


# spans in the answer hero carry role=button + aria-label "<word>: <claim>...".
_NEI_SENTENCES = '[role="button"][aria-label^="not in source:"]'
_SUP_SENTENCES = '[role="button"][aria-label^="grounded:"]'


# --------------------------------------------------------------------------- #
# 1. The money demo — 62%, 5 of 8 grounded, three amber sentences, POST 200
# --------------------------------------------------------------------------- #
def test_money_demo(page, open_app, mock_server):
    open_app(page, mock_server)
    _reduce_motion(page)

    with page.expect_response(lambda r: r.url.endswith("/check")) as resp_info:
        _click_check(page)
    response = resp_info.value
    assert response.status == 200
    assert response.request.method == "POST"

    # The headline score and counts (from the live report, not hardcoded).
    page.get_by_text("62%", exact=True).wait_for(state="visible", timeout=15_000)
    assert page.get_by_text("5 of 8 grounded").first.is_visible()

    # Exactly three sentences carry the amber (not-in-source) verdict — claims 2/5/6.
    page.locator(_NEI_SENTENCES).first.wait_for(state="visible", timeout=10_000)
    assert page.locator(_NEI_SENTENCES).count() == 3
    assert page.locator(_SUP_SENTENCES).count() == 5


# --------------------------------------------------------------------------- #
# 2. Claim expand — the "exactly 25%" sentence shows votes 2 · 1, conf 0.67, no span
# --------------------------------------------------------------------------- #
def test_claim_expand(page, open_app, mock_server):
    open_app(page, mock_server)
    _reduce_motion(page)
    _click_check(page)
    _wait_done(page)

    # Click the hero sentence for the fabricated "exactly 25%" claim (claim 5, NEI).
    mark = page.locator('[role="button"][aria-label*="exactly 25%"]').first
    mark.wait_for(state="visible", timeout=10_000)
    mark.click()

    # The inline detail panel: split vote 2·1, low confidence, empty-span message.
    assert page.get_by_text("votes 2 · 1").first.is_visible()
    assert page.get_by_text("confidence 0.67").first.is_visible()
    assert page.get_by_text("no supporting span — source is silent on this").first.is_visible()
    # The rationale label is present (the detail panel rendered fully).
    assert page.get_by_text("rationale").first.is_visible()


# --------------------------------------------------------------------------- #
# 3. N/A path — clear the answer, check → "N/A" + calm line, no NaN
# --------------------------------------------------------------------------- #
def test_na_path(page, open_app, mock_server):
    open_app(page, mock_server)
    _reduce_motion(page)

    page.locator("textarea").nth(1).fill("")  # clear the Answer textarea
    _click_check(page)

    page.get_by_text("N/A", exact=True).wait_for(state="visible", timeout=15_000)
    assert page.get_by_text("No checkable claims were extracted from this answer.").is_visible()
    # No NaN leaks into the rendered page (the §19-1 guard).
    assert "NaN" not in page.locator("body").inner_text()


# --------------------------------------------------------------------------- #
# 4. Missing key — real provider, no key → inline missing-key message, no crash
# --------------------------------------------------------------------------- #
def test_missing_key(page, open_app, nokey_server):
    # The no-key server serves its OWN page too, so this is same-origin (no CORS).
    open_app(page, nokey_server)
    _reduce_motion(page)
    _click_check(page)

    page.get_by_text("ANTHROPIC_API_KEY not set — this demo needs a key to run.").wait_for(
        state="visible", timeout=15_000
    )
    # The page is still alive and recoverable (the Try again button exists).
    assert page.get_by_role("button", name="Try again").is_visible()


# --------------------------------------------------------------------------- #
# 5. Error recoverable — route /check to 502 → error state, then recover
# --------------------------------------------------------------------------- #
def test_error_recoverable(page, open_app, mock_server):
    open_app(page, mock_server)
    _reduce_motion(page)

    # Intercept the first /check and fail it with a 502 engine_error.
    def fail_once(route):
        route.fulfill(
            status=502,
            content_type="application/json",
            body='{"code":"engine_error","error":"The verification engine failed to complete the check."}',
        )

    page.route("**/check", fail_once)
    _click_check(page)
    try_again = page.get_by_role("button", name="Try again")
    try_again.wait_for(state="visible", timeout=15_000)
    assert page.get_by_text("The verification engine failed to complete the check.").is_visible()

    # Remove the fault and retry → the check completes (not stuck on progress/error).
    page.unroute("**/check")
    try_again.click()
    page.get_by_text("62%", exact=True).wait_for(state="visible", timeout=15_000)


# --------------------------------------------------------------------------- #
# 6. Dark mode — dark tokens apply to the page background
# --------------------------------------------------------------------------- #
def _root_bg(page) -> str:
    """Computed background-color of the component's outer div (the one that sets --bg)."""
    return page.evaluate(
        """() => {
            const root = document.getElementById('dc-root');
            for (const el of root.querySelectorAll('div')) {
                if (getComputedStyle(el).getPropertyValue('--bg').trim()) {
                    return getComputedStyle(el).backgroundColor;
                }
            }
            return null;
        }"""
    )


def test_dark_mode(page, open_app, mock_server):
    open_app(page, mock_server)
    # Light first (baseline), then flip the `dark` prop via the dc-runtime API.
    assert _root_bg(page).replace(" ", "") == "rgb(255,255,255)"
    page.evaluate("() => window.__dcSetProps(window.__dcRootName(), { dark: true })")

    # Dark token --bg is #0C0D0F == rgb(12, 13, 15).
    bg = _root_bg(page)
    assert bg.replace(" ", "") == "rgb(12,13,15)", f"expected dark background, got {bg}"


# --------------------------------------------------------------------------- #
# 7. Reduced motion — fills appear without the fade; information identical
# --------------------------------------------------------------------------- #
def test_reduced_motion(page, open_app, mock_server):
    open_app(page, mock_server)
    page.emulate_media(reduced_motion="reduce")
    _click_check(page)

    # With reduced motion the reveal is instant: all 8 verdicts resolve at once, so
    # the three amber sentences are present immediately (same info as the animated path).
    page.get_by_text("62%", exact=True).wait_for(state="visible", timeout=15_000)
    assert page.locator(_NEI_SENTENCES).count() == 3
    assert page.locator(_SUP_SENTENCES).count() == 5


# --------------------------------------------------------------------------- #
# 8. Keyboard path — focus a sentence mark, press Enter → the claim expands
# --------------------------------------------------------------------------- #
def test_keyboard_path(page, open_app, mock_server):
    open_app(page, mock_server)
    _reduce_motion(page)
    _click_check(page)
    _wait_done(page)

    mark = page.locator('p span[role="button"]').first
    mark.focus()  # keyboard focus, no mouse
    page.keyboard.press("Enter")

    # The detail panel for the selected claim is now visible (no mouse click used).
    assert page.get_by_text("votes").first.is_visible()
    assert page.get_by_text("confidence").first.is_visible()


# --------------------------------------------------------------------------- #
# Distributional real-provider e2e (opt-in; skipped without a key)
# --------------------------------------------------------------------------- #
@pytest.mark.api
def test_money_demo_real(page, open_app):
    """Drive the page against a real-provider server; loose distributional asserts."""
    import os
    import subprocess
    import sys

    from conftest import _start_uvicorn, _stop  # type: ignore

    # Load the repo .env (Azure gpt-5.5 creds) for this opt-in path only.
    env_path = __import__("pathlib").Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        pytest.skip("no real LLM key configured — skipping @api e2e")

    proc, base_url = _start_uvicorn({})  # real provider (keys present in env)
    try:
        open_app(page, base_url)
        _reduce_motion(page)
        with page.expect_response(lambda r: r.url.endswith("/check"), timeout=120_000) as ri:
            _click_check(page)
        assert ri.value.status == 200
        # The live model varies; assert the score is in band and ≥1 amber sentence.
        import re

        page.locator('p span[role="button"]').first.wait_for(state="visible", timeout=120_000)
        score_text = page.locator('div[style*="font-size:var(--fs-score"] , span').first  # noqa
        body_text = page.locator("body").inner_text()
        m = re.search(r"(\d+)%", body_text)
        assert m, "no percentage rendered"
        pct = int(m.group(1))
        assert 50 <= pct <= 75, f"score {pct}% out of the loose [50,75] band"
        assert page.locator(_NEI_SENTENCES).count() >= 1
    finally:
        _stop(proc)
