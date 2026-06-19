"""Gemini-only LangChain model construction and response helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

ModelTier = Literal["fast", "analysis"]
FAST_MODEL = "gemini-3.2-flash"
ANALYSIS_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash"


def model_name(config: dict[str, Any], tier: ModelTier) -> str:
    """Resolve the configured Gemini model for a workload tier."""

    key = "fast_model" if tier == "fast" else "analysis_model"
    default = FAST_MODEL if tier == "fast" else ANALYSIS_MODEL
    return str(config.get(key, default))


def create_chat_model(
    config: dict[str, Any],
    tier: ModelTier = "fast",
    *,
    model_override: str | None = None,
) -> BaseChatModel:
    """Create a Gemini chat model through ``langchain-google-genai``."""

    provider = str(config.get("provider", "google")).lower()
    if provider not in {"google", "gemini"}:
        raise ValueError(
            "Only Gemini via langchain-google-genai is supported; "
            f"received provider={provider!r}"
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = model_override or model_name(config, tier)
    temperature = float(config.get("temperature", 0.2))
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _is_model_unavailable(error: Exception) -> bool:
    """Return whether an API failure indicates an unavailable model endpoint."""

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "model not found",
            "model_not_found",
            "not found for api version",
            "404",
        )
    )


def invoke_text(
    prompt: str,
    config: dict[str, Any],
    system_prompt: str = "You are a rigorous AI research engineer.",
    tier: ModelTier = "fast",
) -> str:
    """Invoke the tier-appropriate Gemini model and return text content."""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    selected_model = model_name(config, tier)
    model = create_chat_model(config, tier)
    try:
        response = model.invoke(messages)
    except Exception as error:
        fallback = str(config.get("fallback_model", DEFAULT_FALLBACK_MODEL))
        if not _is_model_unavailable(error) or fallback == selected_model:
            raise
        response = create_chat_model(
            config,
            tier,
            model_override=fallback,
        ).invoke(messages)
    content = response.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def invoke_json(
    prompt: str,
    config: dict[str, Any],
    tier: ModelTier = "fast",
) -> dict[str, Any]:
    """Invoke Gemini and parse a JSON object from its response."""

    text = invoke_text(
        prompt + "\nReturn valid JSON only, without Markdown fences.",
        config,
        tier=tier,
    )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("The LLM response did not contain a JSON object")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("The LLM response must be a JSON object")
    return parsed
