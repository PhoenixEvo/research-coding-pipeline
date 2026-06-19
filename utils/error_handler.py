"""Retry, classification, and recovery helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, ParamSpec, TypeVar

from utils.llm import invoke_json

LOGGER = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


class ErrorType(str, Enum):
    """Categories used to choose an experiment recovery strategy."""

    QUOTA = "quota"
    OOM = "oom"
    TIMEOUT = "timeout"
    CODE_ERROR = "code_error"
    NETWORK = "network"


class ExperimentExecutionError(RuntimeError):
    """Base class for expected remote experiment failures."""


class KaggleQuotaError(ExperimentExecutionError):
    """Kaggle has no accelerator or submission quota remaining."""


class KaggleOOMError(ExperimentExecutionError):
    """The experiment exceeded available device memory."""


class KaggleTimeoutError(ExperimentExecutionError):
    """The remote experiment did not finish before its deadline."""


def classify_error(exception: BaseException) -> ErrorType:
    """Classify an exception using its type and normalized message."""

    if isinstance(exception, KaggleQuotaError):
        return ErrorType.QUOTA
    if isinstance(exception, KaggleOOMError) or isinstance(exception, MemoryError):
        return ErrorType.OOM
    if isinstance(exception, (KaggleTimeoutError, TimeoutError)):
        return ErrorType.TIMEOUT

    message = str(exception).lower()
    if any(token in message for token in ("quota", "too many requests", "rate limit")):
        return ErrorType.QUOTA
    if any(token in message for token in ("out of memory", "oom", "cuda memory")):
        return ErrorType.OOM
    if any(token in message for token in ("timed out", "timeout", "deadline")):
        return ErrorType.TIMEOUT
    if any(token in message for token in ("network", "connection", "dns", "socket")):
        return ErrorType.NETWORK
    return ErrorType.CODE_ERROR


def wrap_with_retry(
    func: Callable[P, R],
    max_retries: int = 3,
    backoff: bool = True,
) -> Callable[P, R]:
    """Wrap a synchronous callable with bounded retry behavior."""

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                delay = 2**attempt if backoff else 0
                LOGGER.warning(
                    "Call failed; retrying",
                    extra={"attempt": attempt + 1, "delay_seconds": delay, "error": str(exc)},
                )
                if delay:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    return wrapped


def handle_oom(exp_config: dict[str, Any]) -> dict[str, Any]:
    """Return a safer experiment config with its batch size halved."""

    updated = dict(exp_config)
    current = int(updated.get("batch_size", 1))
    updated["batch_size"] = max(1, current // 2)
    updated["recovery_reason"] = "oom_batch_size_reduction"
    return updated


def handle_quota(project_config: dict[str, Any]) -> dict[str, Any]:
    """Return project settings switched from Kaggle to Modal."""

    updated = dict(project_config)
    updated["platform"] = "modal"
    return updated


def llm_debug(error_log: str, llm_config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Ask the configured LangChain model for a conservative config-only fix."""

    if not llm_config or not llm_config.get("enabled", True):
        return None
    prompt = (
        "Diagnose this failed ML experiment. Return JSON with a single `config_patch` "
        "object containing only safe hyperparameter changes. Do not propose source-code edits.\n\n"
        f"{error_log[-8000:]}"
    )
    response = invoke_json(prompt, llm_config, tier="fast")
    patch = response.get("config_patch")
    return patch if isinstance(patch, dict) else None
