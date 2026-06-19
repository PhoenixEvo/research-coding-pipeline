"""Generate copy-paste-ready Codex/Antigravity prompts from pipeline outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = REPOSITORY_ROOT / "outputs"
DEFAULT_CONFIGS_ROOT = REPOSITORY_ROOT / "configs"


def _contextual_prompt(context: str, body: str) -> str:
    """Wrap prompt text in the bridge's stable copy-paste format."""

    return f"# Context: {context}\n{body.strip()}\n# ---"


def _missing_prompt(context: str, filename: str) -> str:
    """Return a formatted missing-artifact message without raising."""

    return _contextual_prompt(
        context,
        f"No {filename} found. Run pipeline first or check outputs/ directory.",
    )


def _empty_outputs_prompt(context: str) -> str:
    """Return a formatted message for a missing or empty outputs directory."""

    return _contextual_prompt(
        context,
        "No output directories found. Run pipeline first or check outputs/ directory.",
    )


def _iteration_number(path: Path) -> int:
    """Extract the final iteration-like number from a directory name."""

    numbers = re.findall(r"\d+", path.name)
    return int(numbers[-1]) if numbers else -1


def _latest_timestamp(path: Path) -> float:
    """Return the newest modification timestamp within a directory."""

    timestamps = [path.stat().st_mtime]
    try:
        timestamps.extend(item.stat().st_mtime for item in path.rglob("*") if item.is_file())
    except OSError:
        return path.stat().st_mtime
    return max(timestamps)


def find_latest_output_dir(outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path | None:
    """Return the latest direct child directory under ``outputs_root``."""

    if not outputs_root.is_dir():
        return None
    candidates = [path for path in outputs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (_latest_timestamp(path), _iteration_number(path), path.name),
    )


def _resolve_output_dir(
    output_dir: Path | None,
    outputs_root: Path,
) -> Path | None:
    """Use an explicit output directory or auto-detect the latest one."""

    return output_dir if output_dir is not None else find_latest_output_dir(outputs_root)


def _load_json(path: Path) -> Any:
    """Load a UTF-8 JSON artifact."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize common experiment artifact shapes to a list of records."""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "experiment_results",
        "results",
        "experiments",
        "exp_configs",
        "configs",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in payload for key in ("config", "metrics", "hyperparameters")):
        return [payload]
    return []


def _record_config(record: dict[str, Any]) -> dict[str, Any]:
    """Extract hyperparameters from a normalized experiment record."""

    for key in ("config", "hyperparameters", "params"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    excluded = {
        "id",
        "metrics",
        "status",
        "platform",
        "duration",
        "error",
        "traceback",
    }
    return {key: value for key, value in record.items() if key not in excluded}


def _record_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Extract numeric metrics from an experiment record."""

    value = record.get("metrics")
    if isinstance(value, dict):
        return {
            str(key): float(metric)
            for key, metric in value.items()
            if isinstance(metric, (int, float)) and not isinstance(metric, bool)
        }
    return {}


def _project_name(output_dir: Path, payload: Any) -> str:
    """Resolve a safe project name from metadata or the output directory."""

    if isinstance(payload, dict):
        for key in ("project_name", "project", "name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value.strip()).stem
    return output_dir.name


def _primary_metric_from_config(project_name: str, configs_root: Path) -> str | None:
    """Read a top-level ``primary_metric`` value without requiring YAML parsing."""

    path = configs_root / f"{project_name}.yaml"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^primary_metric\s*:\s*['\"]?([^'\"#\s]+)", line)
        if match:
            return match.group(1)
    return None


def _primary_metric(
    payload: Any,
    records: list[dict[str, Any]],
    project_name: str,
    configs_root: Path,
) -> str | None:
    """Resolve the primary metric from artifact metadata, config, or metrics."""

    if isinstance(payload, dict):
        value = payload.get("primary_metric")
        if isinstance(value, str) and value:
            return value
        project_config = payload.get("project_config")
        if isinstance(project_config, dict):
            value = project_config.get("primary_metric")
            if isinstance(value, str) and value:
                return value

    configured = _primary_metric_from_config(project_name, configs_root)
    if configured:
        return configured

    metric_names = {
        name for record in records for name in _record_metrics(record)
    }
    for preferred in ("score", "accuracy", "f1_weighted", "psnr"):
        if preferred in metric_names:
            return preferred
    return sorted(metric_names)[0] if metric_names else None


def _best_record(
    records: list[dict[str, Any]],
    primary_metric: str,
) -> dict[str, Any] | None:
    """Select the record with the highest available primary metric."""

    candidates = [
        record
        for record in records
        if primary_metric in _record_metrics(record)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda record: _record_metrics(record)[primary_metric])


def _pretty_json(value: Any) -> str:
    """Serialize prompt context as stable, readable JSON."""

    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def apply_config_prompt(
    output_dir: Path | None = None,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    configs_root: Path = DEFAULT_CONFIGS_ROOT,
) -> str:
    """Generate a prompt that applies the best experiment configuration."""

    context = "Apply the pipeline's best experiment configuration to the project"
    latest = _resolve_output_dir(output_dir, outputs_root)
    if latest is None:
        return _empty_outputs_prompt(context)
    artifact = latest / "exp_configs.json"
    if not artifact.is_file():
        return _missing_prompt(context, "exp_configs.json")
    try:
        payload = _load_json(artifact)
    except (OSError, json.JSONDecodeError) as error:
        return _contextual_prompt(context, f"Could not read exp_configs.json: {error}")

    records = _records_from_payload(payload)
    project_name = _project_name(latest, payload)
    primary_metric = _primary_metric(payload, records, project_name, configs_root)
    if not records or not primary_metric:
        return _contextual_prompt(
            context,
            "No scored experiment configs were found in exp_configs.json. "
            "Run the pipeline first or check outputs/ directory.",
        )
    best = _best_record(records, primary_metric)
    if best is None:
        return _contextual_prompt(
            context,
            f"No experiment contains the primary metric '{primary_metric}'. "
            "Check exp_configs.json.",
        )

    target_config = f"configs/{project_name}.yaml"
    body = f"""Apply the best pipeline configuration below to `{target_config}`.

Primary metric: `{primary_metric}`
Best metric value: `{_record_metrics(best)[primary_metric]}`
Experiment ID: `{best.get("id", "unknown")}`

Best configuration:
```json
{_pretty_json(_record_config(best))}
```

Requirements:
1. Update `{target_config}` with these values while preserving unrelated settings.
2. Inspect `train.py` and every model/training entry point it calls.
3. Ensure every listed hyperparameter is read from the YAML config; remove hardcoded
   duplicates and hidden defaults that override the config.
4. Keep CLI overrides only when they explicitly take precedence and document that order.
5. Add or update focused tests proving the YAML values reach the training/model code.
6. Run the relevant tests and report the changed files and verification results.

Do not change the training algorithm beyond what is required to apply this configuration."""
    return _contextual_prompt(context, body)


def _failed_record_for_error(
    payload: Any,
    error_log: str,
) -> dict[str, Any] | None:
    """Find the experiment record most likely associated with an error log."""

    records = _records_from_payload(payload)
    failed = [
        record
        for record in records
        if str(record.get("status", "")).lower() in {"failed", "error"}
        or record.get("error")
    ]
    for record in reversed(failed):
        experiment_id = str(record.get("id", ""))
        error = str(record.get("error", ""))
        if (experiment_id and experiment_id in error_log) or (
            error and error[:120] in error_log
        ):
            return record
    return failed[-1] if failed else None


def _error_guidance(error_log: str) -> str:
    """Return targeted debugging guidance for common training failures."""

    normalized = error_log.lower()
    hints: list[str] = []
    if "out of memory" in normalized or "oom" in normalized:
        hints.append(
            "OOM detected: verify memory pressure, reduce `batch_size`, and consider "
            "gradient accumulation without changing effective optimization semantics."
        )
    if "cuda" in normalized:
        hints.append(
            "CUDA error detected: check tensor/model device placement, dtype compatibility, "
            "and the exact operation identified by the traceback."
        )
    if "keyerror" in normalized or "key error" in normalized:
        hints.append(
            "KeyError detected: compare config keys with every access site, validate aliases, "
            "and fail early with a clear schema error."
        )
    return "\n".join(f"- {hint}" for hint in hints) or (
        "- Follow the traceback to the first project-owned frame and fix the root cause, "
        "not merely the final exception."
    )


def fix_error_prompt(
    output_dir: Path | None = None,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> str:
    """Generate a root-cause debugging prompt from the latest error artifact."""

    context = "Diagnose and fix the latest failed pipeline experiment"
    latest = _resolve_output_dir(output_dir, outputs_root)
    if latest is None:
        return _empty_outputs_prompt(context)
    error_path = latest / "error_log.txt"
    if not error_path.is_file():
        return _missing_prompt(context, "error_log.txt")
    try:
        error_log = error_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return _contextual_prompt(context, f"Could not read error_log.txt: {error}")

    config_context: dict[str, Any] | None = None
    configs_path = latest / "exp_configs.json"
    if configs_path.is_file():
        try:
            failed = _failed_record_for_error(_load_json(configs_path), error_log)
            if failed:
                config_context = _record_config(failed)
        except (OSError, json.JSONDecodeError):
            config_context = None

    body = f"""Diagnose and fix the root cause of this failed research experiment.

Experiment configuration:
```json
{_pretty_json(config_context) if config_context is not None else "Configuration unavailable in exp_configs.json."}
```

Full error traceback:
```text
{error_log.rstrip()}
```

Relevant diagnostic guidance:
{_error_guidance(error_log)}

Requirements:
1. Trace the failure through the project-owned code and identify the root cause.
2. Inspect the training config, `train.py`, and affected model/data files before editing.
3. Make the smallest robust fix; do not hide failures with broad exception handling.
4. Preserve the intended experiment configuration unless a resource-safe change is required.
5. Add a regression test that reproduces the failure before the fix and passes afterward.
6. Run the relevant tests and summarize the root cause, changed files, and verification."""
    return _contextual_prompt(context, body)


def _suggestion_text(item: Any) -> str | None:
    """Normalize a suggestion entry to text."""

    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, dict):
        for key in ("suggestion", "text", "title", "description"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _top_suggestion(payload: Any) -> str | None:
    """Select the first/highest-priority optimizer suggestion."""

    suggestions: Any = payload
    if isinstance(payload, dict):
        suggestions = payload.get(
            "improvement_suggestions",
            payload.get("suggestions", payload.get("items")),
        )
    if not isinstance(suggestions, list):
        return _suggestion_text(suggestions)

    ranked = sorted(
        enumerate(suggestions),
        key=lambda pair: (
            -float(pair[1].get("priority", 0))
            if isinstance(pair[1], dict)
            and isinstance(pair[1].get("priority"), (int, float))
            else 0,
            pair[0],
        ),
    )
    for _, item in ranked:
        text = _suggestion_text(item)
        if text:
            return text
    return None


def implement_suggestion_prompt(
    output_dir: Path | None = None,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> str:
    """Generate a prompt to implement the optimizer's top suggestion."""

    context = "Implement the optimizer's top improvement suggestion"
    latest = _resolve_output_dir(output_dir, outputs_root)
    if latest is None:
        return _empty_outputs_prompt(context)
    artifact = latest / "improvement_suggestions.json"
    if not artifact.is_file():
        return _missing_prompt(context, "improvement_suggestions.json")
    try:
        suggestion = _top_suggestion(_load_json(artifact))
    except (OSError, json.JSONDecodeError) as error:
        return _contextual_prompt(
            context,
            f"Could not read improvement_suggestions.json: {error}",
        )
    if not suggestion:
        return _contextual_prompt(
            context,
            "No usable suggestion found in improvement_suggestions.json. "
            "Run pipeline first or check outputs/ directory.",
        )

    body = f"""Implement this highest-priority pipeline suggestion:

> {suggestion}

Requirements:
1. Inspect `train.py`, the active model architecture, and project YAML before editing.
2. Identify the smallest coherent implementation that advances the suggestion.
3. Expose new tunable behavior through the project YAML rather than hardcoding it.
4. Preserve backward compatibility for existing configs whenever practical.
5. Add focused tests for configuration plumbing and the changed training/model behavior.
6. Run the relevant tests and summarize implementation choices, changed files, and results.

Do not implement unrelated suggestions from the optimizer in this change."""
    return _contextual_prompt(context, body)


def _heading_level(line: str) -> int:
    """Return a Markdown heading level, or zero for a non-heading."""

    match = re.match(r"^(#{1,6})\s+", line)
    return len(match.group(1)) if match else 0


def _extract_findings(markdown: str) -> str:
    """Extract the most relevant findings section from an analysis report."""

    lines = markdown.splitlines()
    preferred = ("key findings", "top directions", "methodology findings", "findings")
    for label in preferred:
        for index, line in enumerate(lines):
            level = _heading_level(line)
            if level and label in line.lstrip("#").strip().lower():
                collected: list[str] = []
                for candidate in lines[index + 1 :]:
                    candidate_level = _heading_level(candidate)
                    if candidate_level and candidate_level <= level:
                        break
                    collected.append(candidate)
                text = "\n".join(collected).strip()
                if text:
                    return text
    substantive = markdown.strip()
    return substantive[:4000] if substantive else "No key findings were present."


def refactor_for_next_iteration_prompt(
    output_dir: Path | None = None,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> str:
    """Generate a refactoring prompt focused on the analysis findings."""

    context = "Refactor training code to support the next experiment iteration"
    latest = _resolve_output_dir(output_dir, outputs_root)
    if latest is None:
        return _empty_outputs_prompt(context)
    artifact = latest / "analysis_report.md"
    if not artifact.is_file():
        return _missing_prompt(context, "analysis_report.md")
    try:
        report = artifact.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return _contextual_prompt(context, f"Could not read analysis_report.md: {error}")
    findings = _extract_findings(report)

    body = f"""Refactor the training implementation to better support the next iteration's
focus area, based on these pipeline findings:

```markdown
{findings}
```

Requirements:
1. Inspect `train.py`, model architecture files, config loading, and metric logging.
2. Identify structural friction that makes the highlighted experiments difficult or unsafe.
3. Refactor only what improves configurability, isolation, observability, or repeatability
   for the next iteration; preserve current behavior by default.
4. Keep experiment parameters in YAML and validate them at startup.
5. Add tests that lock in existing behavior and cover the new extension points.
6. Run the relevant tests and explain how the refactor supports the findings above.

Avoid speculative model changes that are not supported by the analysis."""
    return _contextual_prompt(context, body)


def generate_all_prompts(
    output_dir: Path | None = None,
    *,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
    configs_root: Path = DEFAULT_CONFIGS_ROOT,
) -> str:
    """Generate all four prompts for one consistently selected output directory."""

    latest = _resolve_output_dir(output_dir, outputs_root)
    if latest is None:
        return _empty_outputs_prompt("Generate all Codex/Antigravity prompts")
    prompts = [
        apply_config_prompt(
            latest,
            outputs_root=outputs_root,
            configs_root=configs_root,
        ),
        fix_error_prompt(latest, outputs_root=outputs_root),
        implement_suggestion_prompt(latest, outputs_root=outputs_root),
        refactor_for_next_iteration_prompt(latest, outputs_root=outputs_root),
    ]
    return "\n\n".join(prompts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bridge command-line interface."""

    parser = argparse.ArgumentParser(
        description="Generate Codex/Antigravity prompts from the latest pipeline output."
    )
    parser.add_argument(
        "command",
        choices=("apply", "fix", "suggest", "refactor", "all"),
        help="Prompt type to generate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the prompt-generator CLI."""

    args = _parse_args(argv)
    generators = {
        "apply": apply_config_prompt,
        "fix": fix_error_prompt,
        "suggest": implement_suggestion_prompt,
        "refactor": refactor_for_next_iteration_prompt,
        "all": generate_all_prompts,
    }
    sys.stdout.write(generators[args.command]())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
