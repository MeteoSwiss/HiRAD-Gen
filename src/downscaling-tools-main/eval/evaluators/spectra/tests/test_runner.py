"""Smoke tests for spectra runner — verify it builds the right argv and invokes subprocess."""
from __future__ import annotations

from subprocess import CalledProcessError
from unittest.mock import patch

import pytest

from eval.evaluators.spectra import runner


def test_run_invokes_predictions_dir_spectra_with_lane_config(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20230828_step024.nc").touch()
    output_dir = tmp_path / "out"
    lane_config = {}
    eval_config = {
        "weather_states": "10u,10v,2t,msl,t_850,z_500",
        "nside": 64,
        "lmax": 319,
        "member_aggregation": "raw-members",
    }

    with patch("eval.evaluators.spectra.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        result = runner.run(
            predictions_dir, lane_config, eval_config,
            output_dir=output_dir, run_label="test_run",
        )

    assert result == output_dir
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "predictions_dir_spectra.py" in cmd[1]
    assert "--predictions-dir" in cmd
    assert str(predictions_dir.resolve()) in cmd
    assert "--out-dir" in cmd
    assert str(output_dir) in cmd
    assert "--run-label" in cmd
    assert "test_run" in cmd
    assert "--weather-states" in cmd
    assert "10u,10v,2t,msl,t_850,z_500" in cmd


def test_run_raises_when_subprocess_fails(tmp_path):
    predictions_dir = tmp_path / "preds"
    predictions_dir.mkdir()
    (predictions_dir / "predictions_20230828_step024.nc").touch()

    with patch("eval.evaluators.spectra.runner.subprocess.run") as mock_run:
        mock_run.side_effect = CalledProcessError(1, ["python"])
        with pytest.raises(CalledProcessError):
            runner.run(predictions_dir, {}, {}, output_dir=tmp_path / "out")
