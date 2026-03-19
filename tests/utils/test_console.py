import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from termcolor import colored

from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper


############################################################################
#                            PythonLogger                                  #
############################################################################


class TestPythonLogger:
    """Tests for PythonLogger."""

    def test_default_name(self):
        pl = PythonLogger()
        assert pl.logger.name == "launch"

    def test_custom_name(self):
        pl = PythonLogger(name="my_trainer")
        assert pl.logger.name == "my_trainer"

    def test_log_calls_info(self, caplog):
        pl = PythonLogger(name="test_log")
        pl.logger.setLevel(logging.DEBUG)
        with caplog.at_level(logging.INFO, logger="test_log"):
            pl.log("hello")
        assert "hello" in caplog.text

    def test_info_uses_light_blue(self, caplog):
        pl = PythonLogger(name="test_info")
        pl.logger.setLevel(logging.DEBUG)
        expected = colored("info msg", "light_blue")
        with caplog.at_level(logging.INFO, logger="test_info"):
            pl.info("info msg")
        assert expected in caplog.text

    def test_success_uses_light_green(self, caplog):
        pl = PythonLogger(name="test_success")
        pl.logger.setLevel(logging.DEBUG)
        expected = colored("ok", "light_green")
        with caplog.at_level(logging.INFO, logger="test_success"):
            pl.success("ok")
        assert expected in caplog.text

    def test_warning_uses_light_yellow(self, caplog):
        pl = PythonLogger(name="test_warning")
        pl.logger.setLevel(logging.DEBUG)
        expected = colored("careful", "light_yellow")
        with caplog.at_level(logging.WARNING, logger="test_warning"):
            pl.warning("careful")
        assert expected in caplog.text

    def test_error_uses_light_red(self, caplog):
        pl = PythonLogger(name="test_error")
        pl.logger.setLevel(logging.DEBUG)
        expected = colored("bad", "light_red")
        with caplog.at_level(logging.ERROR, logger="test_error"):
            pl.error("bad")
        assert expected in caplog.text

    def test_file_logging_creates_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        pl = PythonLogger(name="test_file_create")
        pl.logger.setLevel(logging.DEBUG)
        pl.file_logging(log_file)
        pl.log("file message")
        assert os.path.exists(log_file)
        with open(log_file) as f:
            content = f.read()
        assert "file message" in content

    def test_file_logging_removes_existing_file(self, tmp_path):
        log_file = str(tmp_path / "old.log")
        with open(log_file, "w") as f:
            f.write("old content")
        pl = PythonLogger(name="test_file_overwrite")
        pl.logger.setLevel(logging.DEBUG)
        pl.file_logging(log_file)
        pl.log("new content")
        with open(log_file) as f:
            content = f.read()
        assert "old content" not in content
        assert "new content" in content

    def test_file_logging_formatter(self, tmp_path):
        log_file = str(tmp_path / "fmt.log")
        pl = PythonLogger(name="test_fmt")
        pl.logger.setLevel(logging.DEBUG)
        pl.file_logging(log_file)
        pl.log("fmt check")
        with open(log_file) as f:
            content = f.read()
        # Format: [HH:MM:SS - name - LEVEL] message
        assert "test_fmt" in content
        assert "INFO" in content

    def test_file_logging_level_is_debug(self, tmp_path):
        log_file = str(tmp_path / "debug.log")
        pl = PythonLogger(name="test_debug_level")
        pl.logger.setLevel(logging.DEBUG)
        pl.file_logging(log_file)
        pl.logger.debug("debug msg")
        with open(log_file) as f:
            content = f.read()
        assert "debug msg" in content


############################################################################
#                       RankZeroLoggingWrapper                             #
############################################################################


class TestRankZeroLoggingWrapper:
    """Tests for RankZeroLoggingWrapper."""

    def test_rank_zero_calls_method(self):
        inner = MagicMock()
        inner.log = MagicMock(return_value="logged")
        dist = MagicMock()
        dist.rank = 0

        wrapper = RankZeroLoggingWrapper(inner, dist)
        result = wrapper.log("hello")
        inner.log.assert_called_once_with("hello")
        assert result == "logged"

    def test_non_zero_rank_suppresses_method(self):
        inner = MagicMock()
        inner.log = MagicMock(return_value="logged")
        dist = MagicMock()
        dist.rank = 1

        wrapper = RankZeroLoggingWrapper(inner, dist)
        result = wrapper.log("hello")
        inner.log.assert_not_called()
        assert result is None

    def test_non_callable_attribute_returned_directly(self):
        inner = MagicMock()
        inner.some_value = 42
        dist = MagicMock()
        dist.rank = 5

        wrapper = RankZeroLoggingWrapper(inner, dist)
        assert wrapper.some_value == 42

    def test_rank_zero_passes_kwargs(self):
        inner = MagicMock()
        inner.configure = MagicMock(return_value="configured")
        dist = MagicMock()
        dist.rank = 0

        wrapper = RankZeroLoggingWrapper(inner, dist)
        result = wrapper.configure(level="DEBUG", verbose=True)
        inner.configure.assert_called_once_with(level="DEBUG", verbose=True)
        assert result == "configured"

    def test_multiple_ranks_only_zero_logs(self):
        inner = MagicMock()
        inner.info = MagicMock()

        for rank in range(4):
            dist = MagicMock()
            dist.rank = rank
            wrapper = RankZeroLoggingWrapper(inner, dist)
            wrapper.info("msg")

        # Only rank 0 should have triggered the call
        inner.info.assert_called_once_with("msg")

    def test_wrapper_preserves_return_value(self):
        inner = MagicMock()
        inner.compute = MagicMock(return_value={"loss": 0.5})
        dist = MagicMock()
        dist.rank = 0

        wrapper = RankZeroLoggingWrapper(inner, dist)
        result = wrapper.compute()
        assert result == {"loss": 0.5}

    def test_wrapper_with_python_logger(self):
        pl = PythonLogger(name="test_wrapped")
        pl.logger.setLevel(logging.DEBUG)
        dist = MagicMock()
        dist.rank = 0

        wrapper = RankZeroLoggingWrapper(pl, dist)
        # Should not raise
        wrapper.log("from wrapper")

    def test_wrapper_with_python_logger_non_zero_rank(self, caplog):
        pl = PythonLogger(name="test_wrapped_silent")
        pl.logger.setLevel(logging.DEBUG)
        dist = MagicMock()
        dist.rank = 3

        wrapper = RankZeroLoggingWrapper(pl, dist)
        with caplog.at_level(logging.INFO, logger="test_wrapped_silent"):
            wrapper.log("should not appear")
        assert "should not appear" not in caplog.text
