from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eval.jobs import scoreboard_surface_loss as mod


SURFACE_VARS = list(mod.SURFACE_VARIABLES.keys())


def _write_predictions(
    path,
    *,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weather_states: list[str] | list[bytes],
    lat_hres: np.ndarray,
    area_weight: np.ndarray | None = None,
) -> None:
    data_vars = {
        "y_pred": (
            ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
            y_pred,
        ),
        "y": (
            ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
            y_true,
        ),
        "lat_hres": (("grid_point_hres",), lat_hres),
        "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float64)),
    }
    if area_weight is not None:
        data_vars["area_weight"] = (("grid_point_hres",), area_weight)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": [0],
            "ensemble_member": np.arange(y_pred.shape[1]),
            "grid_point_hres": np.arange(y_pred.shape[2]),
            "weather_state": weather_states,
        },
    )
    ds.to_netcdf(path)


def test_process_predictions_dir_averages_across_members_and_variables(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pred_file = predictions_dir / "predictions_20230826_step024.nc"

    y_pred = np.zeros((1, 2, 2, len(SURFACE_VARS)), dtype=np.float64)
    y_true = np.zeros((1, 2, 2, len(SURFACE_VARS)), dtype=np.float64)
    y_pred[:, 0, :, :] = 1.0
    y_pred[:, 1, :, :] = 3.0

    _write_predictions(
        pred_file,
        y_pred=y_pred,
        y_true=y_true,
        weather_states=SURFACE_VARS,
        lat_hres=np.array([0.0, 60.0], dtype=np.float64),
    )

    result = mod.process_predictions_dir(predictions_dir)

    # Member-wise MSE: member0=1, member1=9 -> mean=5 for every variable.
    assert result["weighted_surface_mse"] == pytest.approx(5.0)
    assert result["weighted_surface_nmse"] is None
    assert result["n_prediction_files"] == 1
    assert result["n_member_samples_per_variable"] == 2
    for var in SURFACE_VARS:
        assert result["variables"][var]["mean_mse"] == pytest.approx(5.0)
        assert result["variables"][var]["n_member_samples"] == 2


def test_process_predictions_dir_uses_area_weight_when_available(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pred_file = predictions_dir / "predictions_20230827_step024.nc"

    # Keep only one ensemble member; script should still work.
    y_pred = np.zeros((1, 1, 2, len(SURFACE_VARS)), dtype=np.float64)
    y_true = np.zeros((1, 1, 2, len(SURFACE_VARS)), dtype=np.float64)
    y_pred[:, :, 0, :] = 1.0
    y_pred[:, :, 1, :] = 3.0

    _write_predictions(
        pred_file,
        y_pred=y_pred,
        y_true=y_true,
        weather_states=[v.encode("utf-8") for v in SURFACE_VARS],
        lat_hres=np.array([0.0, 60.0], dtype=np.float64),
        area_weight=np.array([1.0, 3.0], dtype=np.float64),
    )

    result = mod.process_predictions_dir(predictions_dir)

    # area_weight normalized to [0.25, 0.75] -> 0.25*1^2 + 0.75*3^2 = 7.0
    assert result["weighted_surface_mse"] == pytest.approx(7.0)
    assert result["weighted_surface_nmse"] is None
    assert result["n_member_samples_per_variable"] == 1


def test_process_predictions_dir_reports_truth_std_normalized_nmse(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pred_file = predictions_dir / "predictions_20230828_step024.nc"

    y_true = np.zeros((1, 1, 2, len(SURFACE_VARS)), dtype=np.float64)
    y_true[:, :, 0, :] = 0.0
    y_true[:, :, 1, :] = 2.0
    y_pred = y_true + 1.0

    _write_predictions(
        pred_file,
        y_pred=y_pred,
        y_true=y_true,
        weather_states=SURFACE_VARS,
        lat_hres=np.array([0.0, 60.0], dtype=np.float64),
    )

    result = mod.process_predictions_dir(predictions_dir)

    assert result["weighted_surface_mse"] == pytest.approx(1.0)
    assert result["weighted_surface_nmse"] == pytest.approx(1.0)
    for var in SURFACE_VARS:
        assert result["variables"][var]["truth_std"] == pytest.approx(1.0)
        assert result["variables"][var]["mean_nmse"] == pytest.approx(1.0)


def test_process_predictions_dir_renormalizes_when_surface_var_missing(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    pred_file = predictions_dir / "predictions_20230829_step024.nc"

    available_vars = [var for var in SURFACE_VARS if var != "sp"]
    y_pred = np.zeros((1, 2, 2, len(available_vars)), dtype=np.float64)
    y_true = np.zeros((1, 2, 2, len(available_vars)), dtype=np.float64)
    y_pred[:, 0, :, :] = 1.0
    y_pred[:, 1, :, :] = 3.0

    _write_predictions(
        pred_file,
        y_pred=y_pred,
        y_true=y_true,
        weather_states=available_vars,
        lat_hres=np.array([0.0, 60.0], dtype=np.float64),
    )

    result = mod.process_predictions_dir(predictions_dir)

    assert result["weighted_surface_mse"] == pytest.approx(5.0)
    assert result["available_surface_variables"] == available_vars
    assert result["missing_surface_variables"] == ["sp"]
    assert result["total_weight"] == pytest.approx(sum(mod.SURFACE_VARIABLES[var] for var in available_vars))
    assert sum(entry["normalized_weight"] for entry in result["variables"].values()) == pytest.approx(1.0)
