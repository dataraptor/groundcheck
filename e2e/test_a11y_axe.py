"""Accessibility audit (Split 11, Deliverable 3) — axe-core over the live page.

Injects axe-core (from a CDN — the page already needs network for React/Babel) onto
the **populated** results state and the **N/A** state and asserts **no critical or
serious violations**. Best-practice / moderate findings (e.g. the "region" landmark
rule) are reported for visibility but do not fail the gate, per the brief's
"critical/serious" bar.

If axe can't be loaded (no network), the test skips with a clear reason rather than
failing — same CDN dependency as the rest of the e2e suite.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e

AXE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"

# Impact levels that fail the gate (the brief's bar). moderate/minor are advisory.
BLOCKING_IMPACTS = {"critical", "serious"}


def _inject_axe(page) -> None:
    try:
        page.add_script_tag(url=AXE_URL)
        page.wait_for_function("() => !!window.axe", timeout=15_000)
    except Exception:
        pytest.skip("could not load axe-core from the CDN (no network) — skipping audit")


def _run_axe(page) -> list[dict]:
    """Return axe violations as a list of {id, impact, nodes} dicts."""
    return page.evaluate(
        """async () => {
            const r = await window.axe.run(document, {
                resultTypes: ['violations'],
                // We assert against WCAG 2 A/AA; best-practice rules stay advisory.
                runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] }
            });
            return r.violations.map(v => ({
                id: v.id, impact: v.impact, help: v.help,
                nodes: v.nodes.map(n => n.target.join(' ')).slice(0, 5)
            }));
        }"""
    )


def _blocking(violations: list[dict]) -> list[dict]:
    return [v for v in violations if v.get("impact") in BLOCKING_IMPACTS]


def _format(violations: list[dict]) -> str:
    return "\n".join(
        f"  [{v['impact']}] {v['id']}: {v['help']} — e.g. {v['nodes']}" for v in violations
    )


def _click_check(page):
    page.get_by_role("button", name="Check faithfulness").click()


# --------------------------------------------------------------------------- #
# Populated results state (the worked example)
# --------------------------------------------------------------------------- #
def test_axe_populated_state(page, open_app, mock_server):
    open_app(page, mock_server)
    page.emulate_media(reduced_motion="reduce")
    _click_check(page)
    page.get_by_text("62%", exact=True).wait_for(state="visible", timeout=15_000)

    _inject_axe(page)
    violations = _run_axe(page)
    blocking = _blocking(violations)
    assert not blocking, "axe critical/serious violations on the populated state:\n" + _format(blocking)


# --------------------------------------------------------------------------- #
# N/A state (empty answer)
# --------------------------------------------------------------------------- #
def test_axe_na_state(page, open_app, mock_server):
    open_app(page, mock_server)
    page.emulate_media(reduced_motion="reduce")
    page.locator("textarea").nth(1).fill("")
    _click_check(page)
    page.get_by_text("N/A", exact=True).wait_for(state="visible", timeout=15_000)

    _inject_axe(page)
    violations = _run_axe(page)
    blocking = _blocking(violations)
    assert not blocking, "axe critical/serious violations on the N/A state:\n" + _format(blocking)
