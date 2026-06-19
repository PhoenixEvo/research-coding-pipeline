"""Shared state schema for the research experiment graph."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ExperimentResult(TypedDict):
    """Normalized result returned by every experiment executor."""

    id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    status: str
    platform: str
    duration: float
    error: str | None
    traceback: NotRequired[str]


class ResearchState(TypedDict):
    """State passed between nodes in the LangGraph workflow."""

    project_config: dict[str, Any]
    current_iteration: int
    experiment_configs: list[dict[str, Any]]
    experiment_results: list[ExperimentResult]
    analysis_report: str
    improvement_suggestions: list[str]
    should_continue: bool
    final_report: str
    notification_log: list[str]


def initial_state(project_config: dict[str, Any]) -> ResearchState:
    """Create a complete initial state for a project configuration."""

    return ResearchState(
        project_config=project_config,
        current_iteration=0,
        experiment_configs=[],
        experiment_results=[],
        analysis_report="",
        improvement_suggestions=[],
        should_continue=True,
        final_report="",
        notification_log=[],
    )
