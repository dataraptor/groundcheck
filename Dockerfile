# GroundCheck — Faithfulness Firewall. Serves the API + static demo app same-origin.
#
# Default is MOCK mode: the image runs the 62% money demo with NO API key,
# deterministically and for free. Switch to a live model by setting
# GROUNDCHECK_LLM + the matching credentials at run time (see docker-compose.yml).
#
# Note: the browser UI pulls React/Babel from unpkg.com and the Inter font from
# Google Fonts at runtime, so rendering the page needs internet. The engine and
# API are fully self-contained.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- build stage: install the two packages into a venv ----------------------- #
FROM base AS builder

# Self-contained virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
COPY core/ ./core/
COPY api/ ./api/

# Install core first so api's `groundcheck>=0.1` resolves to the local source
# (pip never reaches PyPI for it), then the API + its server deps.
RUN pip install ./core && pip install ./api

# --- runtime stage: slim image with just the venv + the data the app serves -- #
FROM base AS runtime

# A non-root user — never serve as root.
RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
# Only the files the running service actually reads: the static app and the
# worked-example JSONs. (The Python code already lives in the venv.)
COPY app/ ./app/
COPY core/examples/ ./core/examples/

# Mock by default so the image runs with no key. Point the path resolvers at the
# copied data dirs (the package is installed non-editable, so its __file__-relative
# default would land in site-packages, not here).
ENV GROUNDCHECK_LLM=mock \
    GROUNDCHECK_APP_DIR=/app/app \
    GROUNDCHECK_EXAMPLES_DIR=/app/core/examples \
    PORT=8000

USER app
EXPOSE 8000

# Liveness via the key-free /health route (no curl in slim — use python stdlib).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=4).status==200 else 1)"

# exec-into-shell: expands $PORT *and* makes uvicorn PID 1 so it receives SIGTERM
# directly (clean, fast container shutdown). uvicorn binds 0.0.0.0 so the port is
# reachable from the host.
CMD ["sh", "-c", "exec uvicorn groundcheck_api.main:app --host 0.0.0.0 --port ${PORT}"]
