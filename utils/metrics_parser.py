"""Parsers that normalize experiment logs into numeric metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_from_stdout(log_text: str, metric_names: list[str]) -> dict[str, float]:
    """Extract the last numeric occurrence of each named metric."""

    metrics: dict[str, float] = {}
    for name in metric_names:
        pattern = re.compile(
            rf"(?i)(?:^|[\s,|]){re.escape(name)}\s*[:=]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(log_text)
        if matches:
            metrics[name] = float(matches[-1])
    return metrics


def parse_from_csv(csv_path: str | Path) -> dict[str, float]:
    """Read the final row of a metrics CSV as numeric values."""

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return {}
    return {
        str(key): float(value)
        for key, value in frame.iloc[-1].items()
        if pd.notna(value) and isinstance(value, (int, float))
    }


def parse_from_json(json_path: str | Path) -> dict[str, float]:
    """Read numeric metrics from a JSON object or its `metrics` field."""

    with Path(json_path).open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
        data = data["metrics"]
    if not isinstance(data, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def parse_metrics(path: str | Path) -> dict[str, float]:
    """Auto-detect and parse a CSV or JSON metrics file."""

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return parse_from_csv(path)
    if suffix == ".json":
        return parse_from_json(path)
    raise ValueError(f"Unsupported metrics file extension: {suffix}")
