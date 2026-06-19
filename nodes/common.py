"""Shared node helpers."""

from __future__ import annotations

import copy
from typing import Any

from state import ResearchState


def update_state(state: ResearchState, **changes: Any) -> ResearchState:
    """Return a deep-copied state with selected fields replaced."""

    updated = copy.deepcopy(state)
    updated.update(changes)
    return updated  # type: ignore[typeddict-item]


def metric_specs(project_config: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize metric declarations to name/direction dictionaries."""

    normalized: list[dict[str, str]] = []
    for item in project_config.get("metrics", []):
        if isinstance(item, str):
            normalized.append({"name": item, "direction": "maximize"})
        elif isinstance(item, dict) and item.get("name"):
            normalized.append(
                {
                    "name": str(item["name"]),
                    "direction": str(item.get("direction", "maximize")),
                }
            )
    return normalized


def primary_metric(project_config: dict[str, Any]) -> tuple[str, str]:
    """Return the primary metric name and optimization direction."""

    specs = metric_specs(project_config)
    if not specs:
        return "score", "maximize"
    primary_name = str(project_config.get("primary_metric", specs[0]["name"]))
    match = next((item for item in specs if item["name"] == primary_name), specs[0])
    return match["name"], match["direction"]


def completed_results(state: ResearchState) -> list[dict[str, Any]]:
    """Return successful results that contain at least one metric."""

    return [
        result
        for result in state["experiment_results"]
        if result.get("status") == "completed" and result.get("metrics")
    ]


def best_result(state: ResearchState) -> dict[str, Any] | None:
    """Select the best result according to the configured primary metric."""

    name, direction = primary_metric(state["project_config"])
    candidates = [
        result
        for result in completed_results(state)
        if name in result.get("metrics", {})
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: float(item["metrics"][name]),
        reverse=direction == "maximize",
    )[0]
