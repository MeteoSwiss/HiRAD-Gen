from __future__ import annotations

import sys

import numpy as np
import xarray as xr

from . import load_predict_submodule, stub_build_predictions_dataset


def _make_prediction_dataset(
    *,
    date_value=None,
    init_date=np.datetime64("2023-08-26T00:00:00", "ns"),
    valid_time=None,
) -> xr.Dataset:
    valid_time = init_date + np.timedelta64(24, "h") if valid_time is None else valid_time
    date_data = (
        np.array([init_date], dtype="datetime64[ns]") if date_value is None else np.asarray([date_value])
    )

    ds = xr.Dataset(
        {
            "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), np.ones((1, 2, 3, 2), dtype=np.float32)),
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.ones((1, 2, 4, 2), dtype=np.float32)),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.full((1, 2, 4, 2), 2.0, dtype=np.float32)),
            "date": (("sample",), date_data),
            "init_date": (("sample",), np.array([init_date], dtype="datetime64[ns]")),
            "lead_step_hours": (("sample",), np.array([24], dtype=np.int32)),
            "valid_time": (("sample",), np.array([valid_time], dtype="datetime64[ns]")),
            "lon_hres": (("grid_point_hres",), np.linspace(0.0, 3.0, 4)),
            "lat_hres": (("grid_point_hres",), np.linspace(10.0, 13.0, 4)),
            "lon_lres": (("grid_point_lres",), np.linspace(0.0, 2.0, 3)),
            "lat_lres": (("grid_point_lres",), np.linspace(10.0, 12.0, 3)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [1, 2],
            "grid_point_hres": [0, 1, 2, 3],
            "grid_point_lres": [0, 1, 2],
            "weather_state": ["t2m", "u10"],
        },
    )

    for name, long_name in {
        "date": "forecast initialization date",
        "init_date": "forecast initialization date",
        "valid_time": "forecast valid time",
    }.items():
        ds[name].attrs["long_name"] = long_name
        ds[name].encoding["units"] = "nanoseconds since 1970-01-01 00:00:00"
        ds[name].encoding["calendar"] = "standard"

    ds["lead_step_hours"].attrs["units"] = "hours"
    ds.attrs.update(
        {
            "init_date": "2023-08-26T00:00:00",
            "lead_step_hours": 24,
            "checkpoint_id": "ckpt-1",
        }
    )
    return ds


def test_validate_predictions_dataset_accepts_valid_dataset(monkeypatch):
    """validate_predictions_dataset should return no errors for a valid dataset."""

    mod = load_predict_submodule(monkeypatch, "dataset_builder")
    ds = _make_prediction_dataset()

    assert mod.validate_predictions_dataset(ds) == []



def test_validate_predictions_dataset_reports_missing_variables(monkeypatch):
    """validate_predictions_dataset should report every missing required variable."""

    mod = load_predict_submodule(monkeypatch, "dataset_builder")
    ds = _make_prediction_dataset().drop_vars(["y", "valid_time"])

    errors = mod.validate_predictions_dataset(ds)

    assert "Missing variable: y" in errors
    assert "Missing variable: valid_time" in errors



def test_validate_predictions_dataset_rejects_date_zero_epoch_placeholder(monkeypatch):
    """validate_predictions_dataset should catch the legacy date=0 epoch placeholder bug."""

    mod = load_predict_submodule(monkeypatch, "dataset_builder")
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    ds = _make_prediction_dataset(
        date_value=np.int64(0),
        init_date=epoch,
        valid_time=epoch + np.timedelta64(24, "h"),
    )

    errors = mod.validate_predictions_dataset(ds)

    assert any(error.startswith("date is not datetime64") for error in errors)
    assert "init_date is 1970-01-01T00:00:00 (epoch placeholder)" in errors



def test_build_prediction_dataset_applies_datetime_fix_and_validates(monkeypatch):
    """build_prediction_dataset should stamp real datetimes and pass validation."""

    mod = load_predict_submodule(
        monkeypatch,
        "dataset_builder",
        build_predictions_dataset_fn=stub_build_predictions_dataset,
    )
    types_mod = sys.modules["eval.predict.types"]
    init_date = np.datetime64("2023-08-26T00:00:00", "ns")
    metadata = types_mod.PredictionMetadata(
        init_date=init_date,
        lead_step_hours=24,
        member_ids=[1, 2],
        checkpoint_id="ckpt-1",
        checkpoint_path="/checkpoints/last.ckpt",
        source_bundle_paths=["bundle-1.nc", "bundle-2.nc"],
        sampling_config_json="{}",
        validation_frequency="50h",
        output_weather_state_mode="surface-plus-core-pl",
        output_weather_states=["t2m", "u10"],
        target_members_with_data=2,
        target_members_missing_data=0,
        target_missing_member_ids=[],
        x_interp_exported=False,
        slim_output=True,
    )

    ds = mod.build_prediction_dataset(
        x=np.ones((1, 2, 3, 2), dtype=np.float32),
        y=np.ones((1, 2, 4, 2), dtype=np.float32),
        y_pred=np.full((1, 2, 4, 2), 2.0, dtype=np.float32),
        lon_lres=np.linspace(0.0, 2.0, 3),
        lat_lres=np.linspace(10.0, 12.0, 3),
        lon_hres=np.linspace(0.0, 3.0, 4),
        lat_hres=np.linspace(10.0, 13.0, 4),
        weather_states=["t2m", "u10"],
        init_date=init_date,
        lead_step_hours=24,
        member_ids=[1, 2],
        metadata=metadata,
        source_bundle_paths=["bundle-1.nc", "bundle-2.nc"],
    )

    assert mod.validate_predictions_dataset(ds) == []
    assert ds["date"].dtype == np.dtype("datetime64[ns]")
    assert ds["init_date"].values[0] == init_date
    assert ds["valid_time"].values[0] == init_date + np.timedelta64(24, "h")
    assert ds["date"].encoding["units"] == "nanoseconds since 1970-01-01 00:00:00"
    assert ds.attrs["checkpoint_id"] == "ckpt-1"
