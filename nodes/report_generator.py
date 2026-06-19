"""Create paper-ready Markdown, LaTeX tables, and charts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from nodes.common import best_result, metric_specs, update_state
from state import ResearchState
from utils.chart_generator import (
    plot_comparison_bar,
    plot_hyperparam_sensitivity,
    plot_metric_tradeoff,
    plot_training_curves,
)
from utils.llm import invoke_text
from utils.notifier import notify_pipeline_complete

LOGGER = logging.getLogger(__name__)


def _latex_escape(value: object) -> str:
    """Escape a value for use inside a simple LaTeX table cell."""

    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def _latex_results_table(state: ResearchState) -> str:
    """Render all normalized result metrics as a LaTeX table."""

    metrics = [item["name"] for item in metric_specs(state["project_config"])]
    headers = ["Experiment", "Platform", "Status", *metrics]
    rows = []
    for result in state["experiment_results"]:
        cells = [
            result.get("id", ""),
            result.get("platform", ""),
            result.get("status", ""),
            *[result.get("metrics", {}).get(metric, "--") for metric in metrics],
        ]
        rows.append(" & ".join(_latex_escape(cell) for cell in cells) + r" \\")
    columns = "l" * len(headers)
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(_latex_escape(header) for header in headers) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def _baseline_table(state: ResearchState) -> str:
    """Render configured baselines and the best result as Markdown."""

    metrics = [item["name"] for item in metric_specs(state["project_config"])]
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|---|" + "|".join("---:" for _ in metrics) + "|"
    rows = []
    best = best_result(state)
    if best:
        rows.append(
            "| Best experiment | "
            + " | ".join(str(best.get("metrics", {}).get(name, "--")) for name in metrics)
            + " |"
        )
    for baseline in state["project_config"].get("baselines", []):
        rows.append(
            f"| {baseline.get('name', 'Baseline')} | "
            + " | ".join(
                str(baseline.get("metrics", {}).get(name, "--")) for name in metrics
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _fallback_report(state: ResearchState, chart_paths: list[str]) -> str:
    """Build a complete deterministic final report."""

    best = best_result(state)
    best_config = best.get("config", {}) if best else {}
    return "\n".join(
        [
            f"# {state['project_config'].get('name', 'Research Experiment')} Report",
            "",
            "## Executive summary",
            "",
            f"The pipeline completed {len(state['experiment_results'])} experiments across "
            f"{state['current_iteration']} iteration(s). "
            + (
                f"The best run was `{best['id']}` with metrics "
                f"`{json.dumps(best['metrics'], sort_keys=True)}`."
                if best
                else "No experiment produced a valid primary metric."
            ),
            "",
            "## Best configuration",
            "",
            "```yaml",
            json.dumps(best_config, indent=2, sort_keys=True),
            "```",
            "",
            "## Results table (LaTeX)",
            "",
            "```latex",
            _latex_results_table(state),
            "```",
            "",
            "## Comparison with baselines",
            "",
            _baseline_table(state),
            "",
            "## Experiment analysis",
            "",
            state["analysis_report"],
            "",
            "## Suggested ablation studies",
            "",
            "- Vary one influential hyperparameter at a time around the best run.",
            "- Remove or neutralize each method-specific component.",
            "- Repeat the best configuration across multiple random seeds.",
            "",
            "## Methodology findings",
            "",
            "- Report the full search space, stopping rule, compute platform, and seed policy.",
            "- Distinguish exploratory runs from confirmatory repeated runs.",
            "- Preserve raw logs and configuration snapshots for reproducibility.",
            "",
            "## Limitations",
            "",
            "- Sensitivity estimates are observational and may reflect parameter interactions.",
            "- Remote platform variability can affect runtime and stochastic metrics.",
            "- SOTA comparisons depend on matching datasets, splits, and evaluation protocols.",
            "",
            "## Generated charts",
            "",
            *([f"- `{path}`" for path in chart_paths] or ["- No chart had sufficient data."]),
        ]
    )


def report_generator(state: ResearchState) -> ResearchState:
    """Generate and save the final research report and visualizations."""

    started = time.monotonic()
    config = state["project_config"]
    output_dir = Path(config.get("output_dir", "outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "report_generator start: output_dir=%s results=%s",
        output_dir,
        len(state["experiment_results"]),
    )

    specs = metric_specs(config)
    metric_names = [item["name"] for item in specs]
    chart_paths: list[str] = []
    if len(metric_names) >= 2:
        path = plot_metric_tradeoff(
            state["experiment_results"],
            metric_names[0],
            metric_names[1],
            output_dir / "metric_tradeoff.png",
        )
        if path:
            chart_paths.append(path)
    search_params = list(config.get("search_space", {}))
    if search_params and metric_names:
        path = plot_hyperparam_sensitivity(
            state["experiment_results"],
            search_params[0],
            metric_names[0],
            output_dir / "hyperparameter_sensitivity.png",
        )
        if path:
            chart_paths.append(path)
    path = plot_comparison_bar(
        state["experiment_results"],
        config.get("baselines", []),
        metric_names,
        output_dir / "baseline_comparison.png",
    )
    if path:
        chart_paths.append(path)
    path = plot_training_curves(
        state["experiment_results"],
        output_dir / "training_curves.png",
    )
    if path:
        chart_paths.append(path)

    report = ""
    llm_config = config.get("llm", {})
    if llm_config.get("enabled", True):
        prompt = (
            "Write a concise paper-ready Markdown report with an executive summary, exact best "
            "hyperparameters, comparison against every baseline, ablation suggestions, "
            "methodology findings, and limitations. Include the supplied LaTeX result table "
            "verbatim.\n\n"
            f"Config:\n{json.dumps(config, indent=2)}\n\n"
            f"Results:\n{json.dumps(state['experiment_results'], indent=2)}\n\n"
            f"Analysis:\n{state['analysis_report']}\n\n"
            f"LaTeX table:\n{_latex_results_table(state)}"
        )
        try:
            LOGGER.info("report_generator invoking LLM")
            report = invoke_text(prompt, llm_config, tier="analysis")
        except Exception:
            LOGGER.exception("LLM report generation failed; using deterministic report")
    if not report:
        report = _fallback_report(state, chart_paths)

    report_path = output_dir / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(state["experiment_results"], indent=2, default=str),
        encoding="utf-8",
    )
    notify_pipeline_complete(str(report_path))
    LOGGER.info(
        "report_generator complete: report=%s duration=%.3fs",
        report_path,
        time.monotonic() - started,
    )
    return update_state(
        state,
        final_report=report,
        notification_log=[
            *state["notification_log"],
            f"Pipeline complete: {report_path}",
        ],
    )
