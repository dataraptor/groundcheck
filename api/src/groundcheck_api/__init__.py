"""groundcheck_api — a thin FastAPI HTTP adapter over the ``groundcheck`` engine.

The package holds **no** grounding/scoring/highlighting logic of its own (spec §4):
it validates a request, calls :func:`groundcheck.check`, captures any engine
warnings, maps engine edge cases to clean JSON responses, and serves the static
``app/`` front-end from the same origin (so the page can ``fetch('/check')`` with no
CORS). The ASGI app object is :data:`groundcheck_api.main.app`.
"""

from .main import app

__all__ = ["app"]
