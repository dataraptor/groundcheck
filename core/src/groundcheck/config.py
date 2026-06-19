"""Configuration constants (no logic).

Every number is pulled from the README "API facts" / the spec. Nothing here imports
``anthropic`` or touches the network — keep ``import groundcheck`` key-free and fast.
"""

from __future__ import annotations

from .models import Label

# --- model routing (spec §10; README API facts: exact strings, no date suffix) ---
DECOMPOSE_MODEL = "claude-sonnet-4-6"
GROUND_MODEL = "claude-opus-4-8"

# --- OpenAI / Azure provider (Split 02 divergence resolution) ----------------- #
# The only API key supplied in this environment is Azure OpenAI (gpt-5.5), NOT an
# Anthropic key (see PROGRESS.md "Open divergences"). The provider seam (llm.py)
# therefore ships an OpenAIProvider alongside AnthropicProvider; MockProvider stays
# the default for all no-key tests / CI / the §5 demo. The logical decompose/ground
# routing above remains Anthropic (pinned by the spec); when the OpenAI provider is
# active, a single Azure deployment (gpt-5.5) serves both steps.
OPENAI_MODEL = "gpt-5.5"

# --- pricing, USD *per token* (spec §10; README API facts) -------------------- #
# Source values, per 1M tokens:
#   Opus 4.8   = $5 input  / $25 output
#   Sonnet 4.6 = $3 input  / $15 output
PRICING: dict[str, dict[str, float]] = {
    GROUND_MODEL: {"input": 5e-6, "output": 25e-6},  # Opus 4.8: $5 / $25 per MTok
    DECOMPOSE_MODEL: {"input": 3e-6, "output": 15e-6},  # Sonnet 4.6: $3 / $15 per MTok
    # gpt-5.5 (Azure) list price is NOT in the pinned README API facts — these are
    # documented ESTIMATES so compute_cost() works for the OpenAI path. Treat the
    # OpenAI-path cost figure as approximate (the demo/eval caption should say so).
    OPENAI_MODEL: {"input": 1.25e-6, "output": 10e-6},  # gpt-5.5 estimate: ~$1.25 / ~$10 per MTok
}

# --- cache-bucket multipliers vs the fresh input price (README API facts) ----- #
CACHE_WRITE_MULTIPLIER = 1.25  # cache_creation_input_tokens ≈ 1.25× input
CACHE_READ_MULTIPLIER = 0.1  # cache_read_input_tokens ≈ 0.1× input

# --- N-run majority (spec §9) / UI stepper range (UI spec §B) ----------------- #
DEFAULT_N_RUNS = 3
N_RUNS_MIN = 1
N_RUNS_MAX = 5

# --- decompose call budget (spec §17 truncation guard) ------------------------ #
DECOMPOSE_MAX_TOKENS = 4096

# --- ground call budget (Split 04) -------------------------------------------- #
# A grounding verdict is tiny (label + a one-span + a one-sentence rationale), so a
# small cap is plenty; large supporting spans on big sources still fit comfortably.
GROUND_MAX_TOKENS = 1024

# --- concurrency (spec §10) --------------------------------------------------- #
THREAD_POOL_WORKERS = 6

# --- prompt caching (spec §10; README API facts) ------------------------------ #
# Opus 4.8 cache floor: sources under this won't cache (cache_creation_input_tokens
# == 0, expected, not a bug). Also the gate for "warm one call before fanning out".
SOURCE_CACHE_FLOOR_TOKENS = 4096

# --- oversize input caps (spec §17) ------------------------------------------- #
# A sane demo cap. Exceeding it → warn + truncate, never crash (enforced in a later
# split). These are the gate for the oversize-source / oversize-answer warning path.
MAX_SOURCE_TOKENS = 16000
MAX_ANSWER_TOKENS = 16000

# --- severity ordering: CONTRADICTED > NOT_ENOUGH_INFO > SUPPORTED ------------ #
# Single source of truth for the majority tie-break (Split 04) and worst-verdict
# sentence highlighting (Split 05).
SEVERITY_ORDER: dict[Label, int] = {
    "SUPPORTED": 0,
    "NOT_ENOUGH_INFO": 1,
    "CONTRADICTED": 2,
}

# --- engine HTML fallback colors (spec §8 pastels) ---------------------------- #
# Note: the dc-html app's palette() uses the lower-chroma design tokens from UI spec
# §3.1 (#E7F3EC / #FBF0DA / #FBE7E7) and renders claim-level, so the frontend does
# NOT consume the engine's highlighted_html. The engine keeps the §8 pastels for any
# consumer that does. There is no React/Next.js app in this plan.
VERDICT_COLORS: dict[Label, str] = {
    "SUPPORTED": "#d6f5d6",
    "NOT_ENOUGH_INFO": "#fff3cd",
    "CONTRADICTED": "#f8d7da",
}

# --- human-readable verdict words for the UI/CLI (matches the mockup legend) --- #
VERDICT_WORDS: dict[Label, str] = {
    "SUPPORTED": "grounded",
    "NOT_ENOUGH_INFO": "not in source",
    "CONTRADICTED": "contradicted",
}
