"""Best-effort Telegram notifications for long-running pipelines."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

LOGGER = logging.getLogger(__name__)


def notify(message: str, level: str = "info") -> bool:
    """Send a Telegram message, or silently skip when credentials are absent."""

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        LOGGER.debug("Telegram notification skipped: credentials are not configured")
        return False

    icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
    payload = json.dumps(
        {"chat_id": chat_id, "text": f"{icons.get(level, 'ℹ️')} {message}"}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except Exception as exc:
        LOGGER.warning("Telegram notification failed: %s", exc)
        return False


def notify_experiment_complete(result: dict[str, Any]) -> bool:
    """Notify that one experiment has completed."""

    return notify(
        f"Experiment {result.get('id')} finished with status "
        f"{result.get('status')}: {result.get('metrics', {})}",
        "success" if result.get("status") == "completed" else "warning",
    )


def notify_iteration_complete(
    iteration: int,
    best_result: dict[str, Any] | None,
    analysis_summary: str,
) -> bool:
    """Notify that an optimization iteration has completed."""

    best_id = best_result.get("id") if best_result else "none"
    return notify(
        f"Iteration {iteration} complete. Best experiment: {best_id}\n"
        f"{analysis_summary[:500]}",
        "success",
    )


def notify_pipeline_complete(final_report_path: str) -> bool:
    """Notify that the pipeline and final report are complete."""

    return notify(f"Research pipeline complete. Report: {final_report_path}", "success")


def notify_error(error_type: object, details: str) -> bool:
    """Notify about a recoverable pipeline error."""

    return notify(f"{error_type}: {details[:1000]}", "error")
