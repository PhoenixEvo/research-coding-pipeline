"""Tests for pipeline-to-Codex prompt generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bridge import codex_prompt_gen as bridge


def _write_json(path: Path, payload: object) -> None:
    """Write a JSON test artifact."""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_find_latest_output_dir_uses_latest_artifact_timestamp(tmp_path: Path) -> None:
    """The newest project artifact wins over directory-name ordering."""

    older = tmp_path / "iteration_99"
    newer = tmp_path / "iteration_01"
    older.mkdir()
    newer.mkdir()
    older_file = older / "analysis_report.md"
    newer_file = newer / "analysis_report.md"
    older_file.write_text("old", encoding="utf-8")
    newer_file.write_text("new", encoding="utf-8")
    os.utime(older_file, (1000, 1000))
    os.utime(newer_file, (2000, 2000))

    assert bridge.find_latest_output_dir(tmp_path) == newer


def test_apply_prompt_selects_highest_primary_metric(tmp_path: Path) -> None:
    """The apply prompt includes the best config and inferred project YAML path."""

    outputs = tmp_path / "outputs"
    project = outputs / "ser_cnn_bilstm"
    configs = tmp_path / "configs"
    project.mkdir(parents=True)
    configs.mkdir()
    (configs / "ser_cnn_bilstm.yaml").write_text(
        "primary_metric: f1_weighted\n",
        encoding="utf-8",
    )
    _write_json(
        project / "exp_configs.json",
        [
            {
                "id": "weak",
                "config": {"learning_rate": 0.001, "dropout": 0.5},
                "metrics": {"f1_weighted": 0.72},
            },
            {
                "id": "best",
                "config": {"learning_rate": 0.0001, "dropout": 0.2},
                "metrics": {"f1_weighted": 0.84},
            },
        ],
    )

    prompt = bridge.apply_config_prompt(
        outputs_root=outputs,
        configs_root=configs,
    )

    assert prompt.startswith("# Context:")
    assert "`configs/ser_cnn_bilstm.yaml`" in prompt
    assert "Best metric value: `0.84`" in prompt
    assert '"dropout": 0.2' in prompt
    assert '"dropout": 0.5' not in prompt
    assert prompt.endswith("# ---")


def test_fix_prompt_includes_traceback_config_and_targeted_hints(tmp_path: Path) -> None:
    """The error prompt preserves the full traceback and failed config."""

    project = tmp_path / "outputs" / "project"
    project.mkdir(parents=True)
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "train.py", line 41\n'
        "    value = config['missing']\n"
        "KeyError: 'missing'\n"
        "RuntimeError: CUDA out of memory"
    )
    (project / "error_log.txt").write_text(traceback, encoding="utf-8")
    _write_json(
        project / "exp_configs.json",
        {
            "experiment_results": [
                {
                    "id": "failed-exp",
                    "status": "failed",
                    "error": "CUDA out of memory",
                    "config": {"batch_size": 64, "device": "cuda"},
                    "metrics": {},
                }
            ]
        },
    )

    prompt = bridge.fix_error_prompt(outputs_root=tmp_path / "outputs")

    assert traceback in prompt
    assert '"batch_size": 64' in prompt
    assert "reduce `batch_size`" in prompt
    assert "tensor/model device placement" in prompt
    assert "compare config keys" in prompt


def test_suggestion_prompt_selects_highest_priority_item(tmp_path: Path) -> None:
    """Priority metadata can override list order for structured suggestions."""

    project = tmp_path / "outputs" / "project"
    project.mkdir(parents=True)
    _write_json(
        project / "improvement_suggestions.json",
        [
            {"suggestion": "Try a wider layer.", "priority": 1},
            {"suggestion": "Add gradient clipping.", "priority": 10},
        ],
    )

    prompt = bridge.implement_suggestion_prompt(outputs_root=tmp_path / "outputs")

    assert "Add gradient clipping." in prompt
    assert "Try a wider layer." not in prompt


def test_refactor_prompt_extracts_key_findings_section(tmp_path: Path) -> None:
    """Only the findings section is supplied as the refactoring focus."""

    project = tmp_path / "outputs" / "project"
    project.mkdir(parents=True)
    (project / "analysis_report.md").write_text(
        """# Analysis

## Key Findings

- Batch size controls stability.
- Metric logging needs per-epoch values.

## Limitations

- This sentence should not enter the extracted findings.
""",
        encoding="utf-8",
    )

    prompt = bridge.refactor_for_next_iteration_prompt(
        outputs_root=tmp_path / "outputs"
    )

    assert "Batch size controls stability." in prompt
    assert "Metric logging needs per-epoch values." in prompt
    assert "This sentence should not enter" not in prompt


@pytest.mark.parametrize(
    ("generator", "filename"),
    [
        (bridge.apply_config_prompt, "exp_configs.json"),
        (bridge.fix_error_prompt, "error_log.txt"),
        (bridge.implement_suggestion_prompt, "improvement_suggestions.json"),
        (bridge.refactor_for_next_iteration_prompt, "analysis_report.md"),
    ],
)
def test_missing_artifacts_return_clear_messages(
    tmp_path: Path,
    generator: object,
    filename: str,
) -> None:
    """Every prompt type handles a missing required artifact without raising."""

    project = tmp_path / "outputs" / "project"
    project.mkdir(parents=True)

    prompt = generator(outputs_root=tmp_path / "outputs")  # type: ignore[operator]

    assert (
        f"No {filename} found. Run pipeline first or check outputs/ directory."
        in prompt
    )
    assert prompt.endswith("# ---")


def test_empty_outputs_suggests_running_pipeline(tmp_path: Path) -> None:
    """An empty outputs root produces an actionable message."""

    outputs = tmp_path / "outputs"
    outputs.mkdir()

    prompt = bridge.apply_config_prompt(outputs_root=outputs)

    assert "No output directories found." in prompt
    assert "Run pipeline first" in prompt


def test_all_mode_emits_four_formatted_prompts(tmp_path: Path) -> None:
    """The all command emits every prompt with its own separator."""

    project = tmp_path / "outputs" / "project"
    configs = tmp_path / "configs"
    project.mkdir(parents=True)
    configs.mkdir()
    (configs / "project.yaml").write_text("primary_metric: score\n", encoding="utf-8")
    _write_json(
        project / "exp_configs.json",
        [{"id": "best", "config": {"width": 64}, "metrics": {"score": 1.0}}],
    )
    (project / "error_log.txt").write_text("RuntimeError: failed", encoding="utf-8")
    _write_json(project / "improvement_suggestions.json", ["Increase width."])
    (project / "analysis_report.md").write_text(
        "## Key Findings\n\n- Wider models performed best.",
        encoding="utf-8",
    )

    prompts = bridge.generate_all_prompts(
        outputs_root=tmp_path / "outputs",
        configs_root=configs,
    )

    assert prompts.count("# Context:") == 4
    assert prompts.count("# ---") == 4


def test_cli_dispatches_requested_generator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI command names dispatch to the expected prompt function."""

    monkeypatch.setattr(bridge, "apply_config_prompt", lambda: "APPLY\n# ---")

    assert bridge.main(["apply"]) == 0
    assert capsys.readouterr().out == "APPLY\n# ---\n"
