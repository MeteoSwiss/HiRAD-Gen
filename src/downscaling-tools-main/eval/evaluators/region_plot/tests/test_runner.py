"""Smoke tests for region_plot runner argv construction."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from eval.evaluators.region_plot import runner


def test_run_invokes_plot_regions_module(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20230828_step024.nc").touch()
    output_dir = tmp_path / "out"
    lane_config = {"regions": {"interesting": {"north_atlantic": [-80, 30, 0, 60]}}}
    eval_config = {}

    with patch("eval.evaluators.region_plot.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.run(predictions_dir, lane_config, eval_config, output_dir=output_dir)

    assert result == output_dir
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "-m" in cmd
    assert "eval._backends.region_plotting.plot_regions" in cmd
    assert "--predictions-nc" in cmd
    assert str((predictions_dir / "predictions_20230828_step024.nc").resolve()) in cmd
    assert "--out-dir" in cmd
    assert str(output_dir) in cmd


def test_run_raises_when_no_predictions(tmp_path):
    predictions_dir = tmp_path / "preds_empty"
    predictions_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        runner.run(predictions_dir, {}, {}, output_dir=tmp_path / "out")
