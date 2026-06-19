"""End-to-end and recovery tests using local executor doubles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nodes.experiment_runner import experiment_runner
from nodes.optimizer import optimizer
from pipeline import run_pipeline, save_state
from state import initial_state
from utils.error_handler import KaggleOOMError, KaggleQuotaError, KaggleTimeoutError
from utils.metrics_parser import parse_from_stdout


def _project_config(output_dir: Path) -> dict[str, Any]:
    """Return a small deterministic project config for unit tests."""

    return {
        "name": "Mock research project",
        "platform": "kaggle",
        "output_dir": str(output_dir),
        "experiments_per_iteration": 2,
        "max_concurrency": 2,
        "max_experiments": 4,
        "max_iterations": 4,
        "time_budget_hours": 1,
        "improvement_threshold": 0.001,
        "primary_metric": "accuracy",
        "llm": {"enabled": False},
        "metrics": [
            {"name": "accuracy", "direction": "maximize"},
            {"name": "latency_ms", "direction": "minimize"},
        ],
        "baselines": [
            {"name": "Baseline", "metrics": {"accuracy": 0.7, "latency_ms": 20.0}}
        ],
        "search_space": {
            "learning_rate": {"values": [0.001, 0.01]},
            "batch_size": {"values": [16, 32]},
        },
    }


async def _successful_executor(
    exp_config: dict[str, Any],
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic metrics derived from the candidate config."""

    del project_config
    learning_rate = float(exp_config["learning_rate"])
    batch_size = int(exp_config["batch_size"])
    return {
        "id": exp_config["id"],
        "config": exp_config,
        "metrics": {
            "accuracy": 0.70 + learning_rate * 4 + batch_size / 1000,
            "latency_ms": 40 - batch_size / 2,
        },
        "status": "completed",
        "platform": exp_config.get("_platform", "kaggle"),
        "duration": 0.01,
        "error": None,
    }


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end_with_mock_executors(tmp_path: Path) -> None:
    """The graph loops, stops at budget, persists state, and writes a report."""

    config = _project_config(tmp_path / "outputs")
    config_path = tmp_path / "project.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    final_state = await run_pipeline(
        config_path,
        kaggle_executor=_successful_executor,
        modal_executor=_successful_executor,
    )

    assert final_state["should_continue"] is False
    assert final_state["current_iteration"] == 2
    assert len(final_state["experiment_results"]) == 4
    assert all(item["status"] == "completed" for item in final_state["experiment_results"])
    output_dir = Path(final_state["project_config"]["output_dir"])
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "state.json").exists()
    persisted = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
    assert len(persisted["experiment_results"]) == 4


@pytest.mark.asyncio
async def test_resume_continues_after_last_persisted_node(tmp_path: Path) -> None:
    """Resume dispatches to the node after the saved transition."""

    config = _project_config(tmp_path / "resume-outputs")
    config["max_experiments"] = 1
    config["_last_node"] = "code_generator"
    state = initial_state(config)
    state["current_iteration"] = 1
    state["experiment_configs"] = [
        {
            "id": "resume-exp",
            "_iteration": 1,
            "_platform": "kaggle",
            "learning_rate": 0.01,
            "batch_size": 16,
        }
    ]
    state_path = save_state(state)
    final_state = await run_pipeline(
        "unused.yaml",
        resume_path=state_path,
        kaggle_executor=_successful_executor,
        modal_executor=_successful_executor,
    )

    assert final_state["current_iteration"] == 1
    assert [item["id"] for item in final_state["experiment_results"]] == ["resume-exp"]
    assert final_state["should_continue"] is False


@pytest.mark.asyncio
async def test_oom_retries_with_halved_batch_size(tmp_path: Path) -> None:
    """An OOM retries once with a reduced batch size."""

    calls: list[int] = []

    async def oom_then_success(
        exp_config: dict[str, Any],
        project_config: dict[str, Any],
    ) -> dict[str, Any]:
        del project_config
        calls.append(int(exp_config["batch_size"]))
        if len(calls) == 1:
            raise KaggleOOMError("CUDA out of memory")
        return await _successful_executor(exp_config, {})

    state = initial_state(_project_config(tmp_path))
    state["experiment_configs"] = [
        {
            "id": "oom-test",
            "_iteration": 1,
            "_platform": "kaggle",
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    ]
    result = await experiment_runner(
        state,
        kaggle_executor=oom_then_success,
        modal_executor=_successful_executor,
    )

    assert calls == [32, 16]
    assert result["experiment_results"][0]["status"] == "completed"
    assert result["experiment_results"][0]["config"]["batch_size"] == 16


@pytest.mark.asyncio
async def test_quota_falls_back_to_modal(tmp_path: Path) -> None:
    """A Kaggle quota error routes only that experiment to Modal."""

    async def quota(
        exp_config: dict[str, Any],
        project_config: dict[str, Any],
    ) -> dict[str, Any]:
        del exp_config, project_config
        raise KaggleQuotaError("GPU quota exceeded")

    state = initial_state(_project_config(tmp_path))
    state["experiment_configs"] = [
        {
            "id": "quota-test",
            "_iteration": 1,
            "_platform": "kaggle",
            "learning_rate": 0.001,
            "batch_size": 16,
        }
    ]
    result = await experiment_runner(
        state,
        kaggle_executor=quota,
        modal_executor=_successful_executor,
    )

    experiment = result["experiment_results"][0]
    assert experiment["status"] == "completed"
    assert experiment["platform"] == "modal"
    assert "quota exceeded" in result["notification_log"][0].lower()


@pytest.mark.asyncio
async def test_timeout_becomes_failed_result(tmp_path: Path) -> None:
    """A timeout is recorded and does not escape the runner node."""

    async def timeout(
        exp_config: dict[str, Any],
        project_config: dict[str, Any],
    ) -> dict[str, Any]:
        del exp_config, project_config
        raise KaggleTimeoutError("experiment timed out")

    state = initial_state(_project_config(tmp_path))
    state["experiment_configs"] = [
        {
            "id": "timeout-test",
            "_iteration": 1,
            "_platform": "kaggle",
            "learning_rate": 0.001,
            "batch_size": 16,
        }
    ]
    result = await experiment_runner(
        state,
        kaggle_executor=timeout,
        modal_executor=_successful_executor,
    )

    experiment = result["experiment_results"][0]
    assert experiment["status"] == "failed"
    assert experiment["error"] == "experiment timed out"


def test_optimizer_conditional_decision(tmp_path: Path) -> None:
    """The optimizer continues under budget and stops at the experiment cap."""

    state = initial_state(_project_config(tmp_path))
    state["current_iteration"] = 1
    state["experiment_configs"] = [{"id": "candidate"}]
    state["experiment_results"] = [
        {
            "id": "first",
            "config": {"_iteration": 1},
            "metrics": {"accuracy": 0.75},
            "status": "completed",
            "platform": "kaggle",
            "duration": 1.0,
            "error": None,
        }
    ]
    assert optimizer(state)["should_continue"] is True

    state["experiment_results"] *= 4
    stopped = optimizer(state)
    assert stopped["should_continue"] is False
    assert "maximum experiment count" in stopped["improvement_suggestions"][0]


def test_sample_logs_parse_last_metric_values() -> None:
    """Sample stdout logs produce normalized final metrics."""

    log_path = Path(__file__).parent / "sample_logs" / "3dgs_training.log"
    metrics = parse_from_stdout(
        log_path.read_text(encoding="utf-8"),
        ["psnr", "model_size_mb", "fps"],
    )
    assert metrics == {"psnr": 29.72, "model_size_mb": 31.0, "fps": 118.4}
