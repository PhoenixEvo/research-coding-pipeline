"""
Tests for utils/notifier.py

- test_mock: không cần token thật, mock HTTP call
- test_real: gửi tin nhắn thật lên Telegram (cần TELEGRAM_BOT_TOKEN + CHAT_ID trong .env)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from utils.notifier import (
    notify,
    notify_error,
    notify_experiment_complete,
    notify_iteration_complete,
    notify_pipeline_complete,
)


# ─────────────────────────────────────────
# MOCK TESTS (không cần token thật)
# ─────────────────────────────────────────

class TestNotifyMock:
    """Test notify() logic mà không gọi Telegram thật."""

    def test_skip_when_no_credentials(self):
        """Phải trả False và không crash khi thiếu token."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("TELEGRAM_CHAT_ID", None)
            result = notify("test message")
        assert result is False

    def test_returns_true_on_success(self):
        """Trả True khi HTTP 200."""
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.status = 200

        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "fake_token",
            "TELEGRAM_CHAT_ID": "123456"
        }):
            with patch("urllib.request.urlopen", return_value=mock_response):
                result = notify("pipeline started", level="info")
        assert result is True

    def test_returns_false_on_http_error(self):
        """Trả False (không crash) khi Telegram API lỗi."""
        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "fake_token",
            "TELEGRAM_CHAT_ID": "123456"
        }):
            with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
                result = notify("test", level="error")
        assert result is False

    def test_all_levels_send_correct_icon(self):
        """Mỗi level phải map đúng icon."""
        sent_texts = []

        def capture_request(request, timeout=15):
            import json
            body = json.loads(request.data.decode())
            sent_texts.append(body["text"])
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            return mock_resp

        with patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "fake_token",
            "TELEGRAM_CHAT_ID": "123456"
        }):
            with patch("urllib.request.urlopen", side_effect=capture_request):
                notify("msg", level="info")
                notify("msg", level="warning")
                notify("msg", level="error")
                notify("msg", level="success")

        assert sent_texts[0].startswith("ℹ️")
        assert sent_texts[1].startswith("⚠️")
        assert sent_texts[2].startswith("❌")
        assert sent_texts[3].startswith("✅")


class TestHelperFunctionsMock:
    """Test các helper functions với mock."""

    def setup_method(self):
        self.mock_env = {
            "TELEGRAM_BOT_TOKEN": "fake_token",
            "TELEGRAM_CHAT_ID": "123456"
        }
        self.mock_response = MagicMock()
        self.mock_response.__enter__ = lambda s: s
        self.mock_response.__exit__ = MagicMock(return_value=False)
        self.mock_response.status = 200

    def test_notify_experiment_complete_success(self):
        result_data = {
            "id": "it1_exp2",
            "status": "completed",
            "metrics": {"psnr": 31.4, "model_size_mb": 21.0}
        }
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_experiment_complete(result_data)
        assert result is True

    def test_notify_experiment_complete_failed(self):
        """Experiment fail vẫn gửi notification (level warning)."""
        result_data = {
            "id": "it1_exp3",
            "status": "failed",
            "metrics": None
        }
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_experiment_complete(result_data)
        assert result is True

    def test_notify_iteration_complete(self):
        best = {"id": "it1_exp2", "metrics": {"psnr": 31.4}}
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_iteration_complete(1, best, "Best PSNR so far: 31.4")
        assert result is True

    def test_notify_iteration_complete_no_best(self):
        """Không crash khi best_result là None."""
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_iteration_complete(1, None, "No results yet")
        assert result is True

    def test_notify_pipeline_complete(self):
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_pipeline_complete("outputs/3dgs/report.md")
        assert result is True

    def test_notify_error(self):
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", return_value=self.mock_response):
                result = notify_error("OOMError", "CUDA out of memory on batch 42")
        assert result is True

    def test_notify_error_long_message_truncated(self):
        """Message dài hơn 1000 ký tự phải được truncate."""
        sent_texts = []

        def capture(request, timeout=15):
            import json
            body = json.loads(request.data.decode())
            sent_texts.append(body["text"])
            r = MagicMock()
            r.__enter__ = lambda s: s
            r.__exit__ = MagicMock(return_value=False)
            r.status = 200
            return r

        long_detail = "x" * 2000
        with patch.dict(os.environ, self.mock_env):
            with patch("urllib.request.urlopen", side_effect=capture):
                notify_error("TimeoutError", long_detail)

        # Text được gửi không được vượt quá giới hạn hợp lý
        assert len(sent_texts[0]) < 1100


# ─────────────────────────────────────────
# REAL TEST (cần token thật trong .env)
# Chạy riêng: uv run pytest tests/test_notifier.py -k real -v -s
# ─────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"),
    reason="Telegram credentials not set — skip real test"
)
class TestNotifyReal:
    """Gửi tin nhắn thật lên Telegram. Cần TELEGRAM_BOT_TOKEN + CHAT_ID trong .env."""

    def test_real_notify_info(self):
        result = notify("🧪 [Test] Pipeline notifier hoạt động!", level="info")
        assert result is True

    def test_real_notify_all_levels(self):
        for level in ["info", "warning", "error", "success"]:
            result = notify(f"[Test] Level: {level}", level=level)
            assert result is True

    def test_real_experiment_complete(self):
        result = notify_experiment_complete({
            "id": "test_exp_001",
            "status": "completed",
            "metrics": {"psnr": 31.4, "model_size_mb": 21.0, "fps": 27.0}
        })
        assert result is True

    def test_real_pipeline_complete(self):
        result = notify_pipeline_complete("outputs/3dgs/report.md")
        assert result is True