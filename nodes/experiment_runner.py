"""Execute a generated experiment batch without failing the graph."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from executors.kaggle_executor import run_kaggle_experiment
from executors.modal_executor import run_modal_experiment
from nodes.common import update_state
from state import ResearchState
from utils.error_handler import ErrorType, classify_error, handle_oom
from utils.notifier import notify_error, notify_experiment_complete

LOGGER = logging.getLogger(__name__)
Executor = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


async def _run_one(
    exp_config: dict[str, Any],
    project_config: dict[str, Any],
    kaggle_executor: Executor,
    modal_executor: Executor,
) -> tuple[dict[str, Any], str]:
    """Execute one config with OOM recovery and Kaggle quota fallback."""

    started = time.monotonic()
    platform = str(exp_config.get("_platform", project_config.get("platform", "kaggle")))
    active_config = dict(exp_config)
    executor = kaggle_executor if platform == "kaggle" else modal_executor
    try:
        result = await executor(active_config, project_config)
    except Exception as first_error:
        error_type = classify_error(first_error)
        LOGGER.warning(
            "Experiment %s failed: type=%s error=%s",
            active_config.get("id"),
            error_type.value,
            first_error,
        )
        if error_type == ErrorType.QUOTA and platform == "kaggle":
            platform = "modal"
            active_config["_platform"] = "modal"
            try:
                result = await modal_executor(active_config, project_config)
            except Exception as fallback_error:
                first_error = fallback_error
                error_type = classify_error(fallback_error)
            else:
                return result, (
                    f"{active_config.get('id')}: Kaggle quota exceeded; completed on Modal"
                )
        elif error_type == ErrorType.OOM:
            active_config = handle_oom(active_config)
            try:
                result = await executor(active_config, project_config)
            except Exception as retry_error:
                first_error = retry_error
                error_type = classify_error(retry_error)
            else:
                return result, (
                    f"{active_config.get('id')}: recovered from OOM with batch_size="
                    f"{active_config.get('batch_size')}"
                )

        notify_error(error_type.value, str(first_error))
        result = {
            "id": str(active_config.get("id", "unknown")),
            "config": active_config,
            "metrics": {},
            "status": "failed",
            "platform": platform,
            "duration": time.monotonic() - started,
            "error": str(first_error),
        }
        return result, f"{result['id']}: failed ({error_type.value})"

    return result, f"{result.get('id')}: completed on {result.get('platform', platform)}"


async def experiment_runner(
    state: ResearchState,
    *,
    kaggle_executor: Executor = run_kaggle_experiment,
    modal_executor: Executor = run_modal_experiment,
) -> ResearchState:
    """Run all generated experiments concurrently with bounded parallelism."""

    started = time.monotonic()
    configs = state["experiment_configs"]
    project_config = state["project_config"]
    max_concurrency = max(1, int(project_config.get("max_concurrency", 1)))
    semaphore = asyncio.Semaphore(max_concurrency)
    LOGGER.info(
        "experiment_runner start: experiments=%s concurrency=%s",
        len(configs),
        max_concurrency,
    )

    async def guarded(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        async with semaphore:
            return await _run_one(
                config,
                project_config,
                kaggle_executor,
                modal_executor,
            )

    pairs = await asyncio.gather(*(guarded(config) for config in configs))
    results = [pair[0] for pair in pairs]
    messages = [pair[1] for pair in pairs]
    for result in results:
        notify_experiment_complete(result)
    LOGGER.info(
        "experiment_runner complete: completed=%s failed=%s duration=%.3fs",
        sum(result.get("status") == "completed" for result in results),
        sum(result.get("status") != "completed" for result in results),
        time.monotonic() - started,
    )
    return update_state(
        state,
        experiment_results=[*state["experiment_results"], *results],
        notification_log=[*state["notification_log"], *messages],
    )
