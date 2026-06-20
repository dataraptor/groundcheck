# app: the demo UI

The user-facing front-end: a single-file demo page with no build step. You paste a source
and an answer, click **Check faithfulness**, and the page calls `POST /check` and animates
the result (green/amber/red highlighting, the per-claim table, and the score).

It holds no business logic of its own; it renders what the API returns and handles user
interaction. React and Babel load from a CDN at runtime, so there is nothing to compile.

**Contains:**

- `GroundCheck.dc.html`: the page (markup, styles, and the React UI inline).
- `support.js`: small helpers shared by the page.

**Served by:** `api`, same-origin under `/app`, so the page can `fetch('/check')` with no
CORS. The simplest way to see it is `./scripts/dev.sh` (or `pwsh scripts/dev.ps1`), which
brings up the stack in mock mode and serves the page at <http://127.0.0.1:8000/>.

**Depends on:** `api` (over HTTP, at runtime).
