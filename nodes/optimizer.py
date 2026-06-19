"""Budget and convergence decision node."""

from __future__ import annotations

import logging
import time
from typing import Any

from nodes.common import completed_results, primary_metric, update_state
from state import ResearchState

LOGGER = logging.getLogger(__name__)


def _iteration_best_values(state: ResearchState) -> list[tuple[int, float]]:
    """Return the best primary metric value observed in each iteration."""

    metric, direction = primary_metric(state["project_config"])
    grouped: dict[int, list[float]] = {}
    for result in completed_results(state):
        value = result.get("metrics", {}).get(metric)
        iteration = result.get("config", {}).get("_iteration")
        if value is not None and iteration is not None:
            grouped.setdefault(int(iteration), []).append(float(value))
    chooser = max if direction == "maximize" else min
    return sorted((iteration, chooser(values)) for iteration, values in grouped.items())


def _suggestions(state: ResearchState) -> list[str]:
    """Derive compact suggestions from the analysis and search state."""

    lines = [
        line.lstrip("-0123456789. ").strip()
        for line in state["analysis_report"].splitlines()
        if line.strip().startswith(("-", "1.", "2.", "3."))
    ]
    useful = [line for line in lines if len(line) > 12]
    return useful[-3:] or [
        "Refine the best-performing hyperparameter region.",
        "Run one-variable ablations around the current best config.",
        "Inspect failed experiments before broadening the search.",
    ]


def optimizer(state: ResearchState) -> ResearchState:
    """Continue only while experiment, time, and improvement budgets allow."""

    started = time.monotonic()
    config = state["project_config"]
    count = len(state["experiment_results"])
    max_experiments = int(config.get("max_experiments", count))
    elapsed_hours = float(config.get("_elapsed_seconds", 0.0)) / 3600
    time_budget = float(config.get("time_budget_hours", float("inf")))
    threshold = float(config.get("improvement_threshold", 0.0))
    max_iterations = int(config.get("max_iterations", 10_000))
    iteration_values = _iteration_best_values(state)

    reasons: list[str] = []
    if count >= max_experiments:
        reasons.append("maximum experiment count reached")
    if elapsed_hours >= time_budget:
        reasons.append("time budget reached")
    if state["current_iteration"] >= max_iterations:
        reasons.append("maximum iteration count reached")
    if not state["experiment_configs"]:
        reasons.append("search space exhausted")

    if len(iteration_values) >= 3:
        _, direction = primary_metric(config)
        recent = [value for _, value in iteration_values[-3:]]
        gains = [
            recent[index] - recent[index - 1]
            if direction == "maximize"
            else recent[index - 1] - recent[index]
            for index in (1, 2)
        ]
        if all(gain < threshold for gain in gains):
            reasons.append(
                f"improvement stayed below {threshold:g} for the last two iterations"
            )

    should_continue = not reasons
    suggestions = _suggestions(state) if should_continue else [
        f"Stopped: {reason}" for reason in reasons
    ]
    LOGGER.info(
        "optimizer complete: continue=%s reasons=%s duration=%.3fs",
        should_continue,
        reasons,
        time.monotonic() - started,
    )
    return update_state(
        state,
        should_continue=should_continue,
        improvement_suggestions=suggestions,
    )
