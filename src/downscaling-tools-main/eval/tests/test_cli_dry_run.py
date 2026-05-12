"""Tests for CLI --dry-run mode."""

from __future__ import annotations

import os
import subprocess
import sys


CODE_ROOT = "/home/ecm5702/dev/downscaling-tools"


def _cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_ROOT + ":" + env.get("PYTHONPATH", "")
    return env


def test_cli_evaluate_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "evaluate",
            "--dry-run",
            "--lane", "o96_o320",
            "--predictions-dir", str(tmp_path),
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"lane": "o96_o320"' in result.stdout


def test_cli_run_dry_run():
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "run",
            "--dry-run",
            "--lane", "o96_o320",
            "--checkpoint", "/tmp/test.ckpt",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert '"lane": "o96_o320"' in result.stdout


def test_cli_evaluate_rejects_quaver_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.cli", "evaluate",
            "--dry-run",
            "--lane", "o96_o320",
            "--predictions-dir", str(tmp_path),
            "--only", "quaver",
        ],
        capture_output=True, text=True, env=_cli_env(), cwd=CODE_ROOT,
    )
    assert result.returncode != 0
    assert "Unknown evaluator(s) in --only: ['quaver']" in result.stderr
