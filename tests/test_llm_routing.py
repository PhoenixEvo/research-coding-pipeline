"""Tests for Gemini-only workload routing and availability fallback."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from utils import llm


class FakeModel:
    """Minimal chat-model double that records invocations."""

    def __init__(
        self,
        model: str,
        calls: list[str],
        *,
        unavailable: bool = False,
    ) -> None:
        self.model = model
        self.calls = calls
        self.unavailable = unavailable

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Record the model and return a deterministic response."""

        del messages
        self.calls.append(self.model)
        if self.unavailable:
            raise RuntimeError(f"404 model not found: {self.model}")
        return AIMessage(content='{"ok": true}')


def test_model_name_routes_by_workload_tier() -> None:
    """Fast and analysis workloads resolve to their required defaults."""

    assert llm.model_name({}, "fast") == "gemini-3.2-flash"
    assert llm.model_name({}, "analysis") == "gemini-3.5-flash"


def test_non_gemini_provider_is_rejected() -> None:
    """The model factory does not permit another LLM provider."""

    with pytest.raises(ValueError, match="Only Gemini"):
        llm.create_chat_model({"provider": "anthropic"})


def test_fast_model_not_found_falls_back_to_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable requested fast model retries on the configured Gemini fallback."""

    calls: list[str] = []

    def fake_factory(
        config: dict[str, Any],
        tier: llm.ModelTier = "fast",
        *,
        model_override: str | None = None,
    ) -> FakeModel:
        model = model_override or llm.model_name(config, tier)
        return FakeModel(
            model,
            calls,
            unavailable=model == "gemini-3.2-flash",
        )

    monkeypatch.setattr(llm, "create_chat_model", fake_factory)
    response = llm.invoke_json(
        "Return a test object.",
        {
            "fast_model": "gemini-3.2-flash",
            "fallback_model": "gemini-3.5-flash",
        },
        tier="fast",
    )

    assert response == {"ok": True}
    assert calls == ["gemini-3.2-flash", "gemini-3.5-flash"]


def test_analysis_uses_gemini_35_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis-heavy calls directly use Gemini 3.5 Flash."""

    calls: list[str] = []

    def fake_factory(
        config: dict[str, Any],
        tier: llm.ModelTier = "fast",
        *,
        model_override: str | None = None,
    ) -> FakeModel:
        return FakeModel(model_override or llm.model_name(config, tier), calls)

    monkeypatch.setattr(llm, "create_chat_model", fake_factory)
    llm.invoke_text("Analyze results.", {}, tier="analysis")

    assert calls == ["gemini-3.5-flash"]
