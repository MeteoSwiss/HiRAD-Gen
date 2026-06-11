"""Smoke tests for sigma runner argv construction."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from eval.evaluators.sigma import runner


def test_run_invokes_run_sigma_evaluator_module(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    output_dir = tmp_path / "out"
    eval_config = {"n_samples": 5}
    checkpoint = "/home/ecm5702/scratch/aifs/checkpoint/abcd1234/last.ckpt"

    with patch("eval.evaluators.sigma.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.run(
            predictions_dir, {}, eval_config,
            output_dir=output_dir, checkpoint=checkpoint,
        )

    assert result == output_dir
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "-m" in cmd
    assert "eval._backends.sigma_evaluator.run_sigma_evaluator" in cmd
    assert "--name_exp" in cmd
    assert "abcd1234" in cmd
    assert "--name_ckpt" in cmd
    assert "last.ckpt" in cmd
    assert "--n_samples" in cmd
    assert "5" in cmd


def test_run_raises_without_checkpoint(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run(predictions_dir, {}, {}, output_dir=tmp_path / "out")
