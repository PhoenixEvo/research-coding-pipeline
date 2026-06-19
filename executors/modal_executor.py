"""Modal executor for deployed experiment functions."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from typing import Any


async def run_modal_experiment(
    exp_config: dict[str, Any],
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one experiment through a deployed Modal function."""

    import modal

    started = time.monotonic()
    experiment_id = str(exp_config.get("id") or f"modal-{uuid.uuid4().hex[:10]}")
    modal_config = project_config.get("modal", {})
    app_name = modal_config.get("app_name")
    function_name = modal_config.get("function_name")
    if not app_name or not function_name:
        raise ValueError("project_config.modal.app_name and function_name are required")

    function = modal.Function.from_name(app_name, function_name)
    timeout_seconds = float(
        modal_config.get(
            "timeout_seconds",
            float(project_config.get("time_budget_hours", 8)) * 3600,
        )
    )

    async def invoke() -> Any:
        remote_aio = getattr(function.remote, "aio", None)
        if remote_aio is not None:
            return await remote_aio(exp_config, project_config)
        value = await asyncio.to_thread(function.remote, exp_config, project_config)
        if inspect.isawaitable(value):
            return await value
        return value

    payload = await asyncio.wait_for(invoke(), timeout=timeout_seconds)
    if not isinstance(payload, dict):
        raise TypeError("Modal experiment function must return a dictionary")
    return {
        "id": str(payload.get("id", experiment_id)),
        "config": payload.get("config", exp_config),
        "metrics": payload.get("metrics", {}),
        "status": payload.get("status", "completed"),
        "platform": "modal",
        "duration": float(payload.get("duration", time.monotonic() - started)),
        "error": payload.get("error"),
    }
