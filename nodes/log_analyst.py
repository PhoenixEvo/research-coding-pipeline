"""Analyze accumulated experiment results and baseline gaps."""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from typing import Any

from nodes.common import best_result, completed_results, primary_metric, update_state
from state import ResearchState
from utils.llm import invoke_text
from utils.notifier import notify_iteration_complete

LOGGER = logging.getLogger(__name__)


def _numeric_sensitivity(
    results: list[dict[str, Any]],
    metric: str,
) -> list[tuple[str, float]]:
    """Estimate univariate sensitivity using absolute Pearson correlation."""

    param_names = {
        key
        for result in results
        for key, value in result.get("config", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    scores: list[tuple[str, float]] = []
    for name in param_names:
        pairs = [
            (float(result["config"][name]), float(result["metrics"][metric]))
            for result in results
            if name in result.get("config", {}) and metric in result.get("metrics", {})
        ]
        if len(pairs) < 2:
            continue
        xs, ys = zip(*pairs)
        if statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
            correlation = 0.0
        else:
            mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
            covariance = statistics.mean(
                (x - mean_x) * (y - mean_y) for x, y in pairs
            )
            correlation = covariance / (statistics.pstdev(xs) * statistics.pstdev(ys))
        if math.isfinite(correlation):
            scores.append((name, abs(correlation)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def _fallback_report(state: ResearchState) -> str:
    """Create a useful local analysis when no LLM is configured."""

    results = completed_results(state)
    best = best_result(state)
    metric, direction = primary_metric(state["project_config"])
    lines = [
        f"# Iteration {state['current_iteration']} analysis",
        "",
        f"- Successful experiments: {len(results)} / {len(state['experiment_results'])}",
        f"- Primary objective: `{metric}` ({direction})",
    ]
    if best:
        lines.extend(
            [
                f"- Best experiment: `{best['id']}` with `{metric}="
                f"{best['metrics'].get(metric)}`",
                f"- Best configuration: `{json.dumps(best['config'], sort_keys=True)}`",
            ]
        )
    else:
        lines.append("- No successful result contains the primary metric.")

    sensitivity = _numeric_sensitivity(results, metric)
    lines.extend(["", "## Hyperparameter sensitivity", ""])
    if sensitivity:
        lines.extend(f"- `{name}`: {score:.3f}" for name, score in sensitivity[:5])
    else:
        lines.append("- Insufficient variation for a numeric sensitivity estimate.")

    lines.extend(["", "## Baseline and SOTA gaps", ""])
    best_value = best["metrics"].get(metric) if best else None
    for baseline in state["project_config"].get("baselines", []):
        baseline_value = baseline.get("metrics", {}).get(metric)
        if best_value is not None and baseline_value is not None:
            gap = float(best_value) - float(baseline_value)
            lines.append(f"- {baseline.get('name', 'baseline')}: gap {gap:+.4f} on `{metric}`")
    sota = state["project_config"].get("sota", {})
    sota_value = sota.get("metrics", {}).get(metric) if isinstance(sota, dict) else None
    if best_value is not None and sota_value is not None:
        lines.append(f"- SOTA gap: {float(best_value) - float(sota_value):+.4f} on `{metric}`")

    directions = [name for name, _ in sensitivity[:3]]
    lines.extend(["", "## Top directions for the next iteration", ""])
    if directions:
        lines.extend(
            f"{index}. Refine `{name}` around the strongest observed region."
            for index, name in enumerate(directions, start=1)
        )
    else:
        lines.extend(
            [
                "1. Increase search-space coverage.",
                "2. Verify metric logging and failed runs.",
                "3. Add controlled single-variable ablations.",
            ]
        )
    return "\n".join(lines)


def log_analyst(state: ResearchState) -> ResearchState:
    """Analyze all results with an LLM or deterministic statistics."""

    started = time.monotonic()
    LOGGER.info(
        "log_analyst start: total_results=%s", len(state["experiment_results"])
    )
    report = ""
    llm_config = state["project_config"].get("llm", {})
    if llm_config.get("enabled", True):
        prompt = (
            "Analyze these machine-learning experiments. Produce structured Markdown with: "
            "best configuration, hyperparameter sensitivity, gaps against every baseline and "
            "SOTA, patterns, trends, and exactly three next directions. Respect metric "
            "maximize/minimize directions.\n\n"
            f"Project:\n{json.dumps(state['project_config'], indent=2)}\n\n"
            f"Results:\n{json.dumps(state['experiment_results'], indent=2)}"
        )
        try:
            LOGGER.info("log_analyst invoking LLM")
            report = invoke_text(prompt, llm_config, tier="analysis")
        except Exception:
            LOGGER.exception("LLM analysis failed; using deterministic analysis")
    if not report:
        report = _fallback_report(state)

    best = best_result(state)
    notify_iteration_complete(state["current_iteration"], best, report)
    LOGGER.info(
        "log_analyst complete: report_chars=%s duration=%.3fs",
        len(report),
        time.monotonic() - started,
    )
    return update_state(
        state,
        analysis_report=report,
        notification_log=[
            *state["notification_log"],
            f"Iteration {state['current_iteration']} analysis complete",
        ],
    )
