# api: the HTTP service

A **thin** FastAPI adapter over the `groundcheck` engine. It holds no business logic: it
validates a request, calls `groundcheck.check()`, surfaces any engine warnings, maps every
edge case to a clean JSON body (never a stack trace), and serves the static `app/`
front-end from the **same origin** so the page can `fetch('/check')` with no CORS.

## Install (editable, dev)

Install the engine first, then this layer:

```bash
python -m pip install -e ./core[dev]
python -m pip install -e "./api[dev]"     # quotes: zsh treats [dev] as a glob
```

`api` declares `groundcheck>=0.1`; installing `core` editable first satisfies it from local
source (pip never reaches PyPI).

## Run

Key-free demo (the whole stack works with no API key):

```bash
GROUNDCHECK_LLM=mock uvicorn groundcheck_api.main:app --reload
```

With a real model, drop `GROUNDCHECK_LLM` and provide credentials: either
`ANTHROPIC_API_KEY`, or Azure OpenAI (`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`).
The engine's `get_provider()` auto-detects which credential is present.

Then open <http://localhost:8000/>, which redirects to the app.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/check` | Verify an answer against a source, returning the full `FaithfulnessReport` + `warnings`. |
| GET | `/examples` | The two worked examples, for the front-end to prefill. |
| GET | `/health` | Liveness + prompt/model identity + `mock_mode`. Never touches the network. |
| GET | `/` | Redirect to `/app/GroundCheck.dc.html`. |
| `/app/*` | (static) | The front-end (`GroundCheck.dc.html`, `support.js`), same-origin. |

### `POST /check`

Request:

```json
{ "source": "...", "answer": "...", "n": 3 }
```

- `n` is bounded to the engine's supported range (`N_RUNS_MIN` to `N_RUNS_MAX`, i.e. 1 to
  5). Out of range gives **422** (request validation).
- An empty or whitespace `answer` is **valid** and takes the N/A path
  (`faithfulness_score: null`, `n_claims: 0`), **200**, not an error.

Response (**200**): the engine's `FaithfulnessReport` verbatim, plus `warnings`. That is
`claims[]` (each with `claim, source_sentence, label, supporting_span, rationale, votes,
confidence, refused`), the counts (`n_claims, n_supported, n_contradicted,
n_not_enough_info, n_low_confidence, n_refused`), `faithfulness_score` (float or null),
`cost_usd`, `latency_s`, `prompt_version`, `n_runs`, `highlighted_html`,
`unlocated_sentences`, and `warnings` (surfaced notices, e.g. oversize truncation).

Errors (no stack trace ever reaches the client):

| Status | `code` | When |
|--------|--------|------|
| 422 | (validation) | `n` out of range, or a malformed body. |
| 503 | `missing_api_key` | A real provider was selected but no key is configured. |
| 502 | `engine_error` | Any other engine/SDK failure (full trace logged server-side only). |

### `GET /health`

```json
{ "status": "ok", "prompt_version": "v3", "mock_mode": true,
  "models": { "decompose": "claude-sonnet-4-6", "ground": "claude-opus-4-8" } }
```

## Paths & CORS

- **Same-origin** is the production path: the app is served under `/app/...` on the same
  host as `/check`, so **no CORS is needed**. A dev-only `CORSMiddleware` allowing loopback
  origins (`localhost` / `127.0.0.1`) is included for when the front-end is opened on a
  different local port.
- The `app/` and `core/examples/` directories are resolved **relative to the repo**
  (computed from `__file__`), so `uvicorn` works regardless of the launch directory.
  Override with `GROUNDCHECK_APP_DIR` / `GROUNDCHECK_EXAMPLES_DIR` if needed.

## Architecture

This layer is an adapter, not a place for business logic. It handles requests, validation,
configuration, and serialization, then delegates the real work to `core`. If you deleted
it, the engine would still work; you'd just lose the HTTP interface.

**Depends on:** `core`.

## Tests

```bash
python -m pytest api/tests -q          # key-free (mock mode), via fastapi TestClient
```

The `@pytest.mark.api` smoke needs a real key; it loads the repo-root `.env` and is skipped
if no key is configured. Note: `api/tests` deliberately has **no** `__init__.py` (unlike
`core/tests` / `eval/tests`) so a combined `pytest core/tests api/tests eval/tests` run
doesn't hit pytest's duplicate-`tests`-package import collision.
