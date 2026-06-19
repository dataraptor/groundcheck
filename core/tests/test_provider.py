"""Tests for the provider seam, refusal/truncation handling, and selection.

All no-key: real providers are exercised through injected stub clients (no
network), and the mock provider needs nothing. Env is controlled via monkeypatch
so the suite is deterministic regardless of what is set on the machine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from groundcheck.llm import (
    REFUSAL,
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    Usage,
    get_provider,
)
from groundcheck.models import Decomposition, DecomposedClaim, GroundingVerdict

# --------------------------------------------------------------------------- #
# Stub clients (dependency injection — never the network layer)
# --------------------------------------------------------------------------- #


class _AnthropicMessages:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _StubAnthropicClient:
    def __init__(self, response):
        self.messages = _AnthropicMessages(response)


def _anthropic_response(stop_reason, *, parsed=None, usage=None, refused=False):
    """Build a stub Anthropic response; on refusal reading parsed_output raises."""
    ns = SimpleNamespace(
        stop_reason=stop_reason,
        usage=usage or SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_details=SimpleNamespace(category="safety") if refused else None,
    )
    if refused:
        # Make accessing parsed content blow up, proving the code never reads it.
        class _Boom(type(ns)):
            @property
            def parsed_output(self):  # noqa: D401
                raise AssertionError("parsed_output read on a refusal")

        boom = _Boom(**vars(ns))
        return boom
    ns.parsed_output = parsed
    return ns


class _OpenAICompletions:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _StubOpenAIClient:
    def __init__(self, response):
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=_OpenAICompletions(response)))


def _openai_response(finish_reason, *, parsed=None, refusal=None, usage=None):
    if refusal is not None:
        class _Msg:
            @property
            def parsed(self):
                raise AssertionError("parsed read on a refusal")

        msg = _Msg()
        msg.refusal = refusal  # type: ignore[attr-defined]
    else:
        msg = SimpleNamespace(parsed=parsed, refusal=None)
    choice = SimpleNamespace(finish_reason=finish_reason, message=msg)
    return SimpleNamespace(
        choices=[choice],
        usage=usage or SimpleNamespace(prompt_tokens=20, completion_tokens=8, prompt_tokens_details=None),
    )


_BLOCKS = [{"type": "text", "text": "SOURCE:\nx\n\nCLAIM:\ny"}]


def _verdict():
    return GroundingVerdict(label="SUPPORTED", supporting_span="x", rationale="ok")


# --------------------------------------------------------------------------- #
# AnthropicProvider
# --------------------------------------------------------------------------- #


def test_anthropic_provider_missing_key_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # No injected client and no key → the key check must fire before any network.
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider().parse(
            model="claude-opus-4-8",
            system="s",
            user_blocks=_BLOCKS,
            output_model=GroundingVerdict,
            max_tokens=512,
        )


def test_anthropic_happy_path():
    resp = _anthropic_response("end_turn", parsed=_verdict())
    provider = AnthropicProvider(client=_StubAnthropicClient(resp))
    result = provider.parse(
        model="claude-opus-4-8",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is False
    assert result.truncated is False
    assert isinstance(result.parsed, GroundingVerdict)
    assert result.usage == Usage(input_tokens=10, output_tokens=5)


def test_anthropic_does_not_send_banned_params():
    resp = _anthropic_response("end_turn", parsed=_verdict())
    stub = _StubAnthropicClient(resp)
    AnthropicProvider(client=stub).parse(
        model="claude-opus-4-8",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    sent = stub.messages.calls[0]
    for banned in ("temperature", "top_p", "top_k", "seed", "budget_tokens", "effort", "thinking"):
        assert banned not in sent


def test_refusal_wrapper():
    resp = _anthropic_response("refusal", refused=True)
    provider = AnthropicProvider(client=_StubAnthropicClient(resp))
    result = provider.parse(
        model="claude-opus-4-8",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is True
    assert result.parsed is None
    assert result.stop_reason == "refusal"


def test_truncation_flag():
    resp = _anthropic_response("max_tokens", parsed=Decomposition(claims=[]))
    provider = AnthropicProvider(client=_StubAnthropicClient(resp))
    result = provider.parse(
        model="claude-sonnet-4-6",
        system="s",
        user_blocks=[{"type": "text", "text": "ANSWER:\nx"}],
        output_model=Decomposition,
        max_tokens=16,
    )
    assert result.truncated is True
    assert result.refused is False


# --------------------------------------------------------------------------- #
# OpenAIProvider (Azure gpt-5.5 — the provider that actually runs the agents)
# --------------------------------------------------------------------------- #


def test_openai_provider_missing_key_message(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI"):
        OpenAIProvider().parse(
            model="gpt-5.5",
            system="s",
            user_blocks=_BLOCKS,
            output_model=GroundingVerdict,
            max_tokens=512,
        )


def test_openai_happy_path():
    resp = _openai_response("stop", parsed=_verdict())
    provider = OpenAIProvider(client=_StubOpenAIClient(resp), deployment="gpt-5.5")
    result = provider.parse(
        model="claude-opus-4-8",  # logical model is ignored; deployment serves it
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is False
    assert isinstance(result.parsed, GroundingVerdict)
    assert result.usage == Usage(input_tokens=20, output_tokens=8)


def test_openai_flattens_blocks_and_uses_deployment():
    resp = _openai_response("stop", parsed=_verdict())
    client = _StubOpenAIClient(resp)
    OpenAIProvider(client=client, deployment="gpt-5.5").parse(
        model="claude-opus-4-8",
        system="rubric",
        user_blocks=[
            {"type": "text", "text": "SOURCE:\nABC", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "\n\nCLAIM:\nXYZ"},
        ],
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    sent = client.beta.chat.completions.calls[0]
    assert sent["model"] == "gpt-5.5"
    # Blocks flattened to one user string (cache_control dropped); system separate.
    assert sent["messages"][0] == {"role": "system", "content": "rubric"}
    assert sent["messages"][1] == {"role": "user", "content": "SOURCE:\nABC\n\nCLAIM:\nXYZ"}
    assert "max_completion_tokens" in sent and "max_tokens" not in sent


def test_openai_refusal_wrapper():
    resp = _openai_response("stop", refusal="I can't help with that.")
    provider = OpenAIProvider(client=_StubOpenAIClient(resp), deployment="gpt-5.5")
    result = provider.parse(
        model="gpt-5.5",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is True
    assert result.parsed is None


def test_openai_content_filter_is_refusal():
    resp = _openai_response("content_filter", parsed=None)
    provider = OpenAIProvider(client=_StubOpenAIClient(resp), deployment="gpt-5.5")
    result = provider.parse(
        model="gpt-5.5",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is True


def test_openai_truncation_flag():
    resp = _openai_response("length", parsed=Decomposition(claims=[]))
    provider = OpenAIProvider(client=_StubOpenAIClient(resp), deployment="gpt-5.5")
    result = provider.parse(
        model="gpt-5.5",
        system="s",
        user_blocks=[{"type": "text", "text": "ANSWER:\nx"}],
        output_model=Decomposition,
        max_tokens=16,
    )
    assert result.truncated is True


# --------------------------------------------------------------------------- #
# MockProvider
# --------------------------------------------------------------------------- #


def test_mock_returns_valid_model():
    provider = MockProvider()
    dec = provider.parse(
        model="m",
        system="s",
        user_blocks=[{"type": "text", "text": "ANSWER:\nhello"}],
        output_model=Decomposition,
        max_tokens=512,
    )
    ver = provider.parse(
        model="m",
        system="s",
        user_blocks=_BLOCKS,
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert isinstance(dec.parsed, Decomposition)
    assert isinstance(ver.parsed, GroundingVerdict)
    # Returned instances are real (round-trip validate).
    Decomposition.model_validate(dec.parsed.model_dump())
    GroundingVerdict.model_validate(ver.parsed.model_dump())
    assert dec.refused is False and ver.refused is False


def test_mock_registered_response_by_substring():
    canned = GroundingVerdict(
        label="CONTRADICTED", supporting_span="no effect on stroke risk", rationale="conflict"
    )
    provider = MockProvider()
    provider.register("no effect on stroke risk", canned)
    result = provider.parse(
        model="m",
        system="s",
        user_blocks=[{"type": "text", "text": "SOURCE:\n...\n\nCLAIM:\nHBP has NO EFFECT ON STROKE RISK."}],
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.parsed == canned


def test_mock_forced_refusal_via_sentinel():
    provider = MockProvider({"benign medical claim": REFUSAL})
    result = provider.parse(
        model="m",
        system="s",
        user_blocks=[{"type": "text", "text": "CLAIM:\nThis is a benign medical claim."}],
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is True
    assert result.parsed is None


def test_mock_forced_refusal_via_refuse_when():
    provider = MockProvider(refuse_when=lambda text: "declineme" in text)
    result = provider.parse(
        model="m",
        system="s",
        user_blocks=[{"type": "text", "text": "CLAIM:\ndeclineme"}],
        output_model=GroundingVerdict,
        max_tokens=512,
    )
    assert result.refused is True


def test_mock_callable_responses():
    def responder(output_model, text):
        if output_model is Decomposition:
            return Decomposition(claims=[DecomposedClaim(claim="c", source_sentence="s.")])
        return None

    provider = MockProvider(responder)
    result = provider.parse(
        model="m",
        system="s",
        user_blocks=[{"type": "text", "text": "ANSWER:\nanything"}],
        output_model=Decomposition,
        max_tokens=512,
    )
    assert result.parsed.claims[0].claim == "c"


# --------------------------------------------------------------------------- #
# get_provider selection
# --------------------------------------------------------------------------- #


def test_get_provider_mock_via_env(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_LLM", "mock")
    assert isinstance(get_provider(), MockProvider)


def test_get_provider_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_LLM", "anthropic")
    assert isinstance(get_provider("mock"), MockProvider)
    assert isinstance(get_provider("openai"), OpenAIProvider)


def test_get_provider_unknown_raises(monkeypatch):
    monkeypatch.delenv("GROUNDCHECK_LLM", raising=False)
    with pytest.raises(ValueError, match="Unknown GROUNDCHECK_LLM"):
        get_provider("nope")


def test_get_provider_autodetect_prefers_present_key(monkeypatch):
    monkeypatch.delenv("GROUNDCHECK_LLM", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert isinstance(get_provider(), AnthropicProvider)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    assert isinstance(get_provider(), OpenAIProvider)


def test_get_provider_default_anthropic_when_nothing_set(monkeypatch):
    monkeypatch.delenv("GROUNDCHECK_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert isinstance(get_provider(), AnthropicProvider)
