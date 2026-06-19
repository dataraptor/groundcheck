"""LLM provider seam, cost accounting, and refusal-safe parsing (spec §10, §17).

This module is the *only* place ``core`` talks to a model API — and it does so
behind a small ``LLMProvider`` Protocol so the engine can run three ways:

* ``AnthropicProvider`` — real Claude calls via ``messages.parse`` (spec/README).
* ``OpenAIProvider``    — real Azure OpenAI (``gpt-5.5``) calls. The only API key
  supplied in this environment is Azure OpenAI, not Anthropic, so this is what
  actually runs the agents (see PROGRESS.md "Open divergences"; resolved here).
* ``MockProvider``      — deterministic, key-free, fixture-driven. The default for
  every no-key test, CI, and the §5 worked-example demo.

Import discipline: ``import groundcheck.llm`` must stay key-free and must NOT pull
in ``anthropic``/``openai`` at module top level — both are imported lazily inside
the providers (only when a real call is made without an injected client).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel

from . import config
from .models import Decomposition, GroundingVerdict

# --------------------------------------------------------------------------- #
# Usage + cost
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: int = 0) -> int:
    """Read ``name`` off an attribute object or a dict; ``default`` if missing/None."""
    if obj is None:
        return default
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if value is None else value


@dataclass(frozen=True)
class Usage:
    """Token counts for one model call, split into the three input cost buckets."""

    input_tokens: int = 0  # fresh, full price
    cache_creation_input_tokens: int = 0  # cache write ≈ 1.25× input
    cache_read_input_tokens: int = 0  # cache read ≈ 0.1× input
    output_tokens: int = 0

    @classmethod
    def from_anthropic(cls, usage_obj: Any) -> "Usage":
        """Map an Anthropic ``usage`` object (or dict) onto ``Usage``; missing → 0."""
        return cls(
            input_tokens=_get(usage_obj, "input_tokens"),
            cache_creation_input_tokens=_get(usage_obj, "cache_creation_input_tokens"),
            cache_read_input_tokens=_get(usage_obj, "cache_read_input_tokens"),
            output_tokens=_get(usage_obj, "output_tokens"),
        )

    @classmethod
    def from_openai(cls, usage_obj: Any) -> "Usage":
        """Map an OpenAI ``usage`` object (or dict) onto ``Usage``.

        OpenAI reports ``prompt_tokens`` (incl. any cached) and
        ``prompt_tokens_details.cached_tokens``; there is no separate cache-write
        bucket, so ``cache_creation_input_tokens`` is always 0 and the fresh
        ``input_tokens`` is ``prompt_tokens - cached_tokens``.
        """
        if usage_obj is None:
            return cls()
        prompt = _get(usage_obj, "prompt_tokens")
        completion = _get(usage_obj, "completion_tokens")
        details = (
            usage_obj.get("prompt_tokens_details")
            if isinstance(usage_obj, dict)
            else getattr(usage_obj, "prompt_tokens_details", None)
        )
        cached = _get(details, "cached_tokens")
        return cls(
            input_tokens=max(prompt - cached, 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached,
            output_tokens=completion,
        )


def compute_cost(model_id: str, usage: Usage) -> float:
    """USD for one call: three input buckets + output (README API facts).

    Fresh input at full price, cache-write at ``CACHE_WRITE_MULTIPLIER`` (1.25×),
    cache-read at ``CACHE_READ_MULTIPLIER`` (0.1×), output at the output price.
    Raises ``KeyError`` for an unknown ``model_id``.
    """
    prices = config.PRICING[model_id]  # KeyError on unknown model — intentional
    in_price = prices["input"]
    out_price = prices["output"]
    return (
        usage.input_tokens * in_price
        + usage.cache_creation_input_tokens * in_price * config.CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * in_price * config.CACHE_READ_MULTIPLIER
        + usage.output_tokens * out_price
    )


# --------------------------------------------------------------------------- #
# Refusal sentinel + result wrapper
# --------------------------------------------------------------------------- #

# Register this as a MockProvider response to force a refusal for matching input
# (lets Split 04 exercise the refusal → NEI path without a key).
REFUSAL = object()


@dataclass(frozen=True)
class ParseResult:
    """The outcome of one structured-output call, normalized across providers."""

    parsed: Optional[BaseModel]  # None when the model refused
    usage: Usage
    stop_reason: str  # "end_turn" | "max_tokens" | "refusal" | "stop" | "length" | ...
    refused: bool  # the model declined (stop_reason == "refusal")
    truncated: bool  # the output hit the token cap


# --------------------------------------------------------------------------- #
# Provider seam
# --------------------------------------------------------------------------- #


class LLMProvider(Protocol):
    """A structured-output call: parse a model response into ``output_model``.

    ``user_blocks`` is a list of Anthropic-style content blocks so callers can
    attach ``cache_control`` to the SOURCE block (Split 04). Providers that don't
    speak that shape (OpenAI) flatten the text and rely on automatic caching.
    """

    def parse(
        self,
        *,
        model: str,
        system: str,
        user_blocks: list[dict],
        output_model: type[BaseModel],
        max_tokens: int,
    ) -> ParseResult: ...


class AnthropicProvider:
    """Real Claude calls via ``client.messages.parse`` (README API facts)."""

    def __init__(self, client: Any = None) -> None:
        # Inject a client for tests; otherwise it is created lazily at call time.
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — this needs a key to run. "
                "Set it or use GROUNDCHECK_LLM=mock."
            )
        import anthropic  # lazy: keep `import groundcheck` key-free

        self._client = anthropic.Anthropic()
        return self._client

    def parse(
        self,
        *,
        model: str,
        system: str,
        user_blocks: list[dict],
        output_model: type[BaseModel],
        max_tokens: int,
    ) -> ParseResult:
        client = self._get_client()
        # No temperature/top_p/top_k/seed/budget_tokens (rejected on Opus 4.8); no
        # thinking (default off); no effort (incompatible with parse) — README facts.
        resp = client.messages.parse(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user_blocks}],
            output_format=output_model,
            max_tokens=max_tokens,
        )
        usage = Usage.from_anthropic(getattr(resp, "usage", None))
        stop_reason = getattr(resp, "stop_reason", None) or ""
        # Check stop_reason BEFORE touching parsed content (spec §17): a refusal
        # returns 200 with empty/partial content.
        if stop_reason == "refusal":
            return ParseResult(
                parsed=None, usage=usage, stop_reason="refusal", refused=True, truncated=False
            )
        return ParseResult(
            parsed=getattr(resp, "parsed_output", None),
            usage=usage,
            stop_reason=stop_reason,
            refused=False,
            truncated=stop_reason == "max_tokens",
        )


class OpenAIProvider:
    """Real Azure OpenAI (``gpt-5.5``) calls via ``beta.chat.completions.parse``.

    One Azure deployment serves both the decompose and ground steps; the logical
    Anthropic ``model`` routing is ignored for the API call (there is a single
    deployment) but still used by callers for cost lookup.
    """

    def __init__(self, client: Any = None, deployment: Optional[str] = None) -> None:
        self._client = client
        self._deployment = deployment

    def _resolve_deployment(self) -> str:
        return (
            self._deployment
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("CHAT_LLM_MODEL")
            or config.OPENAI_MODEL
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT are not set — this needs "
                "Azure OpenAI creds to run. Set them or use GROUNDCHECK_LLM=mock."
            )
        import openai  # lazy: keep `import groundcheck` key-free

        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
        )
        return self._client

    @staticmethod
    def _flatten_blocks(user_blocks: list[dict]) -> str:
        """Join the text of Anthropic content blocks (OpenAI auto-caches; drop
        ``cache_control``)."""
        return "".join(b.get("text", "") for b in user_blocks)

    def parse(
        self,
        *,
        model: str,
        system: str,
        user_blocks: list[dict],
        output_model: type[BaseModel],
        max_tokens: int,
    ) -> ParseResult:
        client = self._get_client()
        resp = client.beta.chat.completions.parse(
            model=self._resolve_deployment(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._flatten_blocks(user_blocks)},
            ],
            response_format=output_model,
            max_completion_tokens=max_tokens,
        )
        usage = Usage.from_openai(getattr(resp, "usage", None))
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None) or ""
        message = choice.message
        # Check refusal/content-filter BEFORE reading parsed content (spec §17).
        if getattr(message, "refusal", None) or finish == "content_filter":
            return ParseResult(
                parsed=None, usage=usage, stop_reason="refusal", refused=True, truncated=False
            )
        return ParseResult(
            parsed=getattr(message, "parsed", None),
            usage=usage,
            stop_reason=finish or "stop",
            refused=False,
            truncated=finish == "length",
        )


# --------------------------------------------------------------------------- #
# Mock provider (deterministic, key-free, fixture-driven)
# --------------------------------------------------------------------------- #

# Fixed small usage for every mock call (keeps demo cost stable and non-zero).
_MOCK_USAGE = Usage(input_tokens=10, output_tokens=10)


def _normalize(text: str) -> str:
    """Lowercase + whitespace-collapsed form used for mock key matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _default_instance(output_model: type[BaseModel]) -> BaseModel:
    """A schema-valid default for an unregistered request.

    Known contracts get a sensible default (empty decomposition / NEI verdict);
    anything else is built generically from its fields so the mock never crashes.
    """
    if output_model is Decomposition:
        return Decomposition(claims=[])
    if output_model is GroundingVerdict:
        return GroundingVerdict(
            label="NOT_ENOUGH_INFO",
            supporting_span="",
            rationale="(mock default) source is silent on this claim.",
        )
    # Generic fallback: fill each required field with a type-appropriate default.
    values: dict[str, Any] = {}
    for name, field in output_model.model_fields.items():
        if not field.is_required():
            continue
        values[name] = _default_for_annotation(field.annotation)
    return output_model(**values)


def _default_for_annotation(annotation: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if annotation in (str,):
        return ""
    if annotation in (int,):
        return 0
    if annotation in (float,):
        return 0.0
    if annotation in (bool,):
        return False
    if origin in (list,):
        return []
    if origin in (dict,):
        return {}
    # Literal[...] → first allowed value (covers the Label enum).
    if args and all(isinstance(a, str) for a in args):
        return args[0]
    return None


class MockProvider:
    """Deterministic provider for tests/CI/demo.

    Register canned responses keyed on a substring of the (normalized) user text:
    the first registered key that is a substring of the request's flattened user
    text wins. A registered value of :data:`REFUSAL` forces a refusal for matching
    input. ``responses`` may instead be a callable
    ``(output_model, normalized_text) -> BaseModel | REFUSAL | None`` for full
    control; ``refuse_when(normalized_text) -> bool`` forces a global refusal rule.
    """

    def __init__(
        self,
        responses: Optional[dict[str, Any] | Callable[[type[BaseModel], str], Any]] = None,
        *,
        refuse_when: Optional[Callable[[str], bool]] = None,
        seed_worked_example: Optional[bool] = None,
    ) -> None:
        self._callable = responses if callable(responses) else None
        self._responses: dict[str, Any] = {} if self._callable else dict(responses or {})
        self._refuse_when = refuse_when
        # Auto-seed the §5 worked-example fixtures into a bare MockProvider so the
        # no-key demo (`cli check --example`) and tests reproduce the canonical 62%
        # run. Default: seed only when the caller passed no explicit `responses`
        # (so tests that supply their own dict/callable keep full control).
        if seed_worked_example is None:
            seed_worked_example = responses is None
        if seed_worked_example:
            self._seed_worked_example()

    def _seed_worked_example(self) -> None:
        """Register the §5 worked-example responses (decomposition here; Split 04
        adds verdicts). Imported lazily to avoid a module-load import cycle."""
        from . import worked_example

        self.register(worked_example.WORKED_EXAMPLE_KEY, worked_example.WORKED_EXAMPLE_DECOMPOSITION)

    def register(self, key: str, value: Any) -> None:
        """Register a canned response (or :data:`REFUSAL`) for inputs containing ``key``."""
        self._responses[_normalize(key)] = value

    def _lookup(self, output_model: type[BaseModel], text: str) -> Any:
        if self._callable is not None:
            return self._callable(output_model, text)
        for key, value in self._responses.items():
            if key not in text:
                continue
            # REFUSAL forces a refusal for any matching request; a typed response
            # only answers a request for *its own* output_model, so a registered
            # Decomposition can't satisfy a GroundingVerdict request even when the
            # claim text overlaps the answer (Split 04). On a type mismatch, keep
            # scanning later keys.
            if value is REFUSAL or isinstance(value, output_model):
                return value
        return None

    def parse(
        self,
        *,
        model: str,
        system: str,
        user_blocks: list[dict],
        output_model: type[BaseModel],
        max_tokens: int,
    ) -> ParseResult:
        text = _normalize("".join(b.get("text", "") for b in user_blocks))
        if self._refuse_when is not None and self._refuse_when(text):
            return ParseResult(
                parsed=None, usage=_MOCK_USAGE, stop_reason="refusal", refused=True, truncated=False
            )
        value = self._lookup(output_model, text)
        if value is REFUSAL:
            return ParseResult(
                parsed=None, usage=_MOCK_USAGE, stop_reason="refusal", refused=True, truncated=False
            )
        parsed = value if isinstance(value, BaseModel) else _default_instance(output_model)
        return ParseResult(
            parsed=parsed, usage=_MOCK_USAGE, stop_reason="end_turn", refused=False, truncated=False
        )


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #

_MOCK_NAMES = {"mock"}
_ANTHROPIC_NAMES = {"anthropic", "claude"}
_OPENAI_NAMES = {"openai", "azure", "azure-openai", "azure_openai", "gpt"}


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """Pick a provider.

    Selection order: explicit ``name`` arg → ``GROUNDCHECK_LLM`` env →
    auto-detect by which credentials are present. ``GROUNDCHECK_LLM=mock`` always
    returns :class:`MockProvider` (the spec/README contract). With no explicit
    choice the active key wins (Azure OpenAI here), falling back to
    :class:`AnthropicProvider` so the missing-key message is the clear one.
    """
    sel = (name or os.getenv("GROUNDCHECK_LLM") or "").strip().lower()
    if sel in _MOCK_NAMES:
        return MockProvider()
    if sel in _ANTHROPIC_NAMES:
        return AnthropicProvider()
    if sel in _OPENAI_NAMES:
        return OpenAIProvider()
    if sel:
        raise ValueError(
            f"Unknown GROUNDCHECK_LLM value {sel!r}; "
            "expected one of: mock, anthropic, openai."
        )
    # No explicit choice → use whichever real credential is configured.
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return OpenAIProvider()
    return AnthropicProvider()
