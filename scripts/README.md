# scripts: run the stack and the suite (one command each)

Cross-platform helpers for bringing up GroundCheck and running its full no-key test suite.
The PowerShell (`*.ps1`, this repo's primary env) and POSIX (`*.sh`) variants are kept in
lock-step. No Docker required to use these (there is also a `docker-compose.yml` at the repo
root if you'd rather run the stack in a container).

## Bring up the stack (mock mode, no API key)

```powershell
# Windows / PowerShell
./scripts/dev.ps1                 # http://127.0.0.1:8000/
./scripts/dev.ps1 -Port 8137
./scripts/dev.ps1 -NoInstall      # skip the editable install
```

```bash
# POSIX
./scripts/dev.sh                  # http://127.0.0.1:8000/
./scripts/dev.sh 8137
NO_INSTALL=1 ./scripts/dev.sh
```

This installs `core` and `api` editable, sets `GROUNDCHECK_LLM=mock`, and serves the
FastAPI app: `POST /check`, `GET /examples`, `GET /health`, and the static page at
`/app/GroundCheck.dc.html` (served same-origin, so the page fetches `/check` with no
CORS). Open `http://127.0.0.1:8000/` and click **Check faithfulness** to see the money demo
(62%, three amber sentences) end-to-end with no key.

## Run the whole no-key suite

```powershell
./scripts/test.ps1                # core + api + eval + e2e + axe audit
./scripts/test.ps1 -SkipE2E       # skip the browser layer
./scripts/test.ps1 -NoInstall
```

```bash
./scripts/test.sh
SKIP_E2E=1 ./scripts/test.sh
NO_INSTALL=1 ./scripts/test.sh
```

Each layer runs with `-m "not api"` so the optional live-key smokes skip. The script
exits non-zero if **any** suite fails.

### Browser e2e (one-time setup)

The `e2e/` Playwright suite needs a browser, and (because the page loads React/Babel from
`unpkg.com` at runtime) **network access for the CDN**. Install the browser once:

```bash
python -m pip install pytest-playwright
python -m playwright install chromium
```

Without chromium the e2e tests **skip** with a clear message (the no-key gate never depends
on them). On an air-gapped box they also skip, because the page can't mount without the CDN,
and the suite reports that explicitly rather than timing out opaquely.

The optional `@pytest.mark.api` real-provider e2e (`test_money_demo_real`) needs a live
key and is skipped without one.
