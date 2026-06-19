"""Kaggle kernel executor with asynchronous status polling."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from utils.error_handler import KaggleOOMError, KaggleQuotaError, KaggleTimeoutError
from utils.metrics_parser import parse_from_stdout, parse_metrics

LOGGER = logging.getLogger(__name__)


def _kernel_status_name(status: Any) -> str:
    """Normalize the various status shapes returned by Kaggle clients."""

    if isinstance(status, str):
        return status.lower()
    for attribute in ("status", "value"):
        value = getattr(status, attribute, None)
        if value:
            return str(value).lower()
    return str(status).lower()


def _raise_specific_kaggle_error(message: str) -> None:
    """Raise a typed exception based on a Kaggle error message."""

    normalized = message.lower()
    if "quota" in normalized or "429" in normalized:
        raise KaggleQuotaError(message)
    if "out of memory" in normalized or "oom" in normalized:
        raise KaggleOOMError(message)
    raise RuntimeError(message)


def _prepare_kernel_source(
    exp_config: dict[str, Any],
    project_config: dict[str, Any],
    working_dir: Path,
    kernel_ref: str,
) -> None:
    """Copy the configured kernel source and write experiment metadata."""

    kaggle_config = project_config.get("kaggle", {})
    source_dir_value = kaggle_config.get("kernel_source_dir")
    if source_dir_value:
        source_dir = Path(source_dir_value).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Kaggle kernel source does not exist: {source_dir}")
        shutil.copytree(source_dir, working_dir, dirs_exist_ok=True)

    with (working_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(exp_config, handle, sort_keys=False)

    username, slug = kernel_ref.split("/", maxsplit=1)
    metadata = {
        "id": kernel_ref,
        "title": slug.replace("-", " ").title(),
        "code_file": kaggle_config.get("code_file", "kernel.py"),
        "language": "python",
        "kernel_type": "script",
        "is_private": bool(kaggle_config.get("is_private", True)),
        "enable_gpu": bool(kaggle_config.get("enable_gpu", True)),
        "enable_internet": bool(kaggle_config.get("enable_internet", False)),
        "dataset_sources": kaggle_config.get("dataset_sources", []),
        "competition_sources": kaggle_config.get("competition_sources", []),
        "kernel_sources": kaggle_config.get("kernel_sources", []),
    }
    del username
    with (working_dir / "kernel-metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _collect_metrics(
    output_dir: Path,
    project_config: dict[str, Any],
) -> dict[str, float]:
    """Collect metrics from the configured output file or downloaded logs."""

    kaggle_config = project_config.get("kaggle", {})
    metrics_file = kaggle_config.get("metrics_file")
    if metrics_file:
        candidate = output_dir / metrics_file
        if candidate.exists():
            return parse_metrics(candidate)

    for name in ("metrics.json", "results.json", "metrics.csv", "results.csv"):
        candidate = output_dir / name
        if candidate.exists():
            return parse_metrics(candidate)

    metric_names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in project_config.get("metrics", [])
    ]
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".log", ".txt", ".out"}
    )
    return parse_from_stdout(log_text, metric_names)


async def run_kaggle_experiment(
    exp_config: dict[str, Any],
    project_config: dict[str, Any],
) -> dict[str, Any]:
    """Push, poll, and download one Kaggle kernel experiment."""

    from kaggle.api.kaggle_api_extended import KaggleApi

    started = time.monotonic()
    experiment_id = str(exp_config.get("id") or f"kaggle-{uuid.uuid4().hex[:10]}")
    kaggle_config = project_config.get("kaggle", {})
    username = kaggle_config.get("username")
    if not username:
        raise ValueError("project_config.kaggle.username is required")
    kernel_slug = f"{kaggle_config.get('kernel_slug_prefix', 'research-exp')}-{experiment_id}"
    kernel_ref = f"{username}/{kernel_slug}".lower().replace("_", "-")
    poll_seconds = float(kaggle_config.get("poll_interval_seconds", 300))
    timeout_seconds = float(
        kaggle_config.get(
            "timeout_seconds",
            float(project_config.get("time_budget_hours", 8)) * 3600,
        )
    )

    api = KaggleApi()
    await asyncio.to_thread(api.authenticate)
    with tempfile.TemporaryDirectory(prefix="research-kaggle-") as temp_dir:
        root = Path(temp_dir)
        source_dir = root / "source"
        output_dir = root / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        _prepare_kernel_source(exp_config, project_config, source_dir, kernel_ref)
        try:
            await asyncio.to_thread(api.kernels_push, str(source_dir))
        except Exception as exc:
            _raise_specific_kaggle_error(str(exc))

        while True:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                raise KaggleTimeoutError(
                    f"Kaggle experiment {experiment_id} exceeded {timeout_seconds} seconds"
                )
            try:
                status = _kernel_status_name(
                    await asyncio.to_thread(api.kernels_status, kernel_ref)
                )
            except Exception as exc:
                _raise_specific_kaggle_error(str(exc))

            LOGGER.info("Kaggle experiment %s status: %s", experiment_id, status)
            if any(token in status for token in ("complete", "success")):
                break
            if any(token in status for token in ("error", "failed", "cancel")):
                _raise_specific_kaggle_error(f"Kaggle kernel ended with status {status}")
            await asyncio.sleep(poll_seconds)

        await asyncio.to_thread(
            api.kernels_output,
            kernel_ref,
            path=str(output_dir),
            force=True,
            quiet=True,
        )
        metrics = _collect_metrics(output_dir, project_config)
    return {
        "id": experiment_id,
        "config": exp_config,
        "metrics": metrics,
        "status": "completed",
        "platform": "kaggle",
        "duration": time.monotonic() - started,
        "error": None,
    }
