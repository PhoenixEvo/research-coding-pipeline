"""Plotly chart generation for experiment reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _write_figure(figure: go.Figure, output_path: str | Path) -> str:
    """Write a figure to PNG, falling back to interactive HTML."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.write_image(str(path))
        return str(path)
    except Exception:
        html_path = path.with_suffix(".html")
        figure.write_html(str(html_path), include_plotlyjs="cdn")
        return str(html_path)


def plot_metric_tradeoff(
    results: list[dict[str, Any]],
    x_metric: str,
    y_metric: str,
    output_path: str | Path,
) -> str | None:
    """Create a scatter plot for two result metrics."""

    rows = [
        {
            "id": item.get("id"),
            x_metric: item.get("metrics", {}).get(x_metric),
            y_metric: item.get("metrics", {}).get(y_metric),
        }
        for item in results
        if x_metric in item.get("metrics", {}) and y_metric in item.get("metrics", {})
    ]
    if not rows:
        return None
    figure = px.scatter(
        pd.DataFrame(rows),
        x=x_metric,
        y=y_metric,
        hover_name="id",
        title=f"{y_metric} vs {x_metric}",
    )
    return _write_figure(figure, output_path)


def plot_hyperparam_sensitivity(
    results: list[dict[str, Any]],
    param_name: str,
    metric: str,
    output_path: str | Path,
) -> str | None:
    """Plot mean metric grouped by a hyperparameter value."""

    rows = [
        {
            param_name: item.get("config", {}).get(param_name),
            metric: item.get("metrics", {}).get(metric),
        }
        for item in results
        if param_name in item.get("config", {}) and metric in item.get("metrics", {})
    ]
    if not rows:
        return None
    frame = pd.DataFrame(rows).groupby(param_name, as_index=False)[metric].mean()
    figure = px.bar(
        frame,
        x=param_name,
        y=metric,
        title=f"{metric} sensitivity to {param_name}",
    )
    return _write_figure(figure, output_path)


def plot_comparison_bar(
    results: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    metrics: list[str],
    output_path: str | Path,
) -> str | None:
    """Compare the best completed experiment with configured baselines."""

    rows: list[dict[str, Any]] = []
    completed = [item for item in results if item.get("status") == "completed"]
    if completed:
        latest_best = completed[-1]
        for metric in metrics:
            value = latest_best.get("metrics", {}).get(metric)
            if value is not None:
                rows.append({"method": latest_best.get("id"), "metric": metric, "value": value})
    for baseline in baselines:
        for metric in metrics:
            value = baseline.get("metrics", {}).get(metric)
            if value is not None:
                rows.append(
                    {"method": baseline.get("name", "baseline"), "metric": metric, "value": value}
                )
    if not rows:
        return None
    figure = px.bar(
        pd.DataFrame(rows),
        x="metric",
        y="value",
        color="method",
        barmode="group",
        title="Experiment and baseline comparison",
    )
    return _write_figure(figure, output_path)


def plot_training_curves(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> str | None:
    """Plot per-step metric histories stored in result `history` arrays."""

    rows: list[dict[str, Any]] = []
    for result in results:
        history = result.get("history")
        if not isinstance(history, list):
            continue
        for index, point in enumerate(history):
            if not isinstance(point, dict):
                continue
            step = point.get("step", point.get("epoch", index + 1))
            for metric, value in point.items():
                if metric not in {"step", "epoch"} and isinstance(value, (int, float)):
                    rows.append(
                        {
                            "experiment": result.get("id", "experiment"),
                            "step": step,
                            "metric": metric,
                            "value": value,
                        }
                    )
    if not rows:
        return None
    figure = px.line(
        pd.DataFrame(rows),
        x="step",
        y="value",
        color="experiment",
        facet_row="metric",
        title="Training curves",
    )
    return _write_figure(figure, output_path)
