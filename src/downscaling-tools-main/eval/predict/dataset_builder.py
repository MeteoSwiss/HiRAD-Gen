"""Dataset assembly helpers with fixed forecast-date handling."""

from __future__ import annotations

import inspect
from typing import Sequence

import numpy as np
import xarray as xr

from manual_inference.prediction.dataset import build_predictions_dataset as _build_predictions_dataset

from .types import PredictionMetadata

_TIME_UNITS = "nanoseconds since 1970-01-01 00:00:00"
_TIME_CALENDAR = "standard"


def _ensure_datetime64_ns(value: np.datetime64) -> np.datetime64:
    return np.asarray(value, dtype="datetime64[ns]").reshape(()).astype("datetime64[ns]")


def _apply_cf_time_metadata(variable: xr.DataArray, *, long_name: str) -> None:
    variable.attrs["long_name"] = long_name
    variable.encoding["units"] = _TIME_UNITS
    variable.encoding["calendar"] = _TIME_CALENDAR


def build_prediction_dataset(
    *,
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    lon_lres: np.ndarray,
    lat_lres: np.ndarray,
    lon_hres: np.ndarray,
    lat_hres: np.ndarray,
    weather_states: Sequence[str],
    init_date: np.datetime64,
    lead_step_hours: int,
    member_ids: Sequence[int],
    x_interp: np.ndarray | None = None,
    include_member_views: bool = True,
    metadata: PredictionMetadata | None = None,
    source_bundle_paths: Sequence[str] | None = None,
) -> xr.Dataset:
    """Build a predictions dataset using a real forecast init date."""

    init_date_dt64 = _ensure_datetime64_ns(init_date)
    build_kwargs = {
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "lon_lres": lon_lres,
        "lat_lres": lat_lres,
        "lon_hres": lon_hres,
        "lat_hres": lat_hres,
        "weather_states": list(weather_states),
        "dates": [init_date_dt64],
        "member_ids": list(member_ids),
        "include_member_views": include_member_views,
    }
    if x_interp is not None and "x_interp" in inspect.signature(_build_predictions_dataset).parameters:
        build_kwargs["x_interp"] = x_interp
    ds = _build_predictions_dataset(**build_kwargs)
    ds = ds.assign_coords(sample=[0])

    valid_time = (init_date_dt64 + np.timedelta64(int(lead_step_hours), "h")).astype("datetime64[ns]")
    ds["date"] = xr.DataArray(np.array([init_date_dt64], dtype="datetime64[ns]"), dims=("sample",))
    ds["init_date"] = xr.DataArray(np.array([init_date_dt64], dtype="datetime64[ns]"), dims=("sample",))
    ds["lead_step_hours"] = xr.DataArray(np.array([int(lead_step_hours)], dtype=np.int32), dims=("sample",))
    ds["valid_time"] = xr.DataArray(np.array([valid_time], dtype="datetime64[ns]"), dims=("sample",))

    _apply_cf_time_metadata(ds["date"], long_name="forecast initialization date")
    _apply_cf_time_metadata(ds["init_date"], long_name="forecast initialization date")
    _apply_cf_time_metadata(ds["valid_time"], long_name="forecast valid time")
    ds["lead_step_hours"].attrs.update({"long_name": "forecast lead step", "units": "hours"})

    ds.attrs["init_date"] = np.datetime_as_string(init_date_dt64, unit="s")
    ds.attrs["lead_step_hours"] = int(lead_step_hours)

    if source_bundle_paths is not None:
        source_paths = [str(path) for path in source_bundle_paths]
        if len(source_paths) != len(member_ids):
            raise ValueError(
                "source_bundle_paths length does not match member_ids length: "
                f"{len(source_paths)} != {len(member_ids)}"
            )
        ds["source_bundle"] = xr.DataArray(
            np.array([source_paths], dtype=object),
            dims=("sample", "ensemble_member"),
        )
        ds["source_bundle"].attrs["long_name"] = "source input bundle path per ensemble member"

    if metadata is not None:
        ds.attrs.update(metadata.to_attrs())
        if metadata.missing_target_policy:
            ds.attrs["missing_target_policy"] = metadata.missing_target_policy

    return ds


def _lead_step_hours_value(array: xr.DataArray) -> int:
    value = np.asarray(array.values).reshape(-1)[0]
    if np.issubdtype(array.dtype, np.timedelta64):
        return int(value / np.timedelta64(1, "h"))
    return int(value)



def validate_predictions_dataset(ds: xr.Dataset) -> list[str]:
    """Return validation errors for the predictions dataset schema."""

    errors: list[str] = []
    required_vars = [
        "x",
        "y",
        "y_pred",
        "date",
        "init_date",
        "lead_step_hours",
        "valid_time",
        "lon_hres",
        "lat_hres",
        "lon_lres",
        "lat_lres",
    ]
    required_dims = ["sample", "ensemble_member", "grid_point_hres", "weather_state"]
    required_attrs = ["init_date", "lead_step_hours", "checkpoint_id"]
    for name in required_vars:
        if name not in ds:
            errors.append(f"Missing variable: {name}")
    for name in required_dims:
        if name not in ds.dims:
            errors.append(f"Missing dimension: {name}")
    for name in required_attrs:
        if name not in ds.attrs:
            errors.append(f"Missing attribute: {name}")

    for time_name in ("date", "init_date", "valid_time"):
        if time_name not in ds:
            continue
        array = ds[time_name]
        if not np.issubdtype(array.dtype, np.datetime64):
            errors.append(f"{time_name} is not datetime64: {array.dtype}")
        if "long_name" not in array.attrs:
            errors.append(f"{time_name} missing long_name metadata")
        units = array.attrs.get("units") or array.encoding.get("units")
        calendar = array.attrs.get("calendar") or array.encoding.get("calendar")
        if units is None:
            errors.append(f"{time_name} missing CF units metadata")
        if calendar is None:
            errors.append(f"{time_name} missing CF calendar metadata")

    if "lead_step_hours" in ds:
        lead_step = ds["lead_step_hours"]
        units = lead_step.attrs.get("units") or lead_step.encoding.get("units")
        if units is None:
            errors.append("lead_step_hours missing units metadata")
        if not (
            np.issubdtype(lead_step.dtype, np.integer) or np.issubdtype(lead_step.dtype, np.timedelta64)
        ):
            errors.append(f"lead_step_hours has unsupported dtype: {lead_step.dtype}")

    if "init_date" in ds and "valid_time" in ds and "lead_step_hours" in ds:
        init_value = np.asarray(ds["init_date"].values).reshape(-1)[0]
        valid_value = np.asarray(ds["valid_time"].values).reshape(-1)[0]
        step_value = _lead_step_hours_value(ds["lead_step_hours"])
        epoch = np.datetime64("1970-01-01T00:00:00", "ns")
        if np.issubdtype(np.asarray(init_value).dtype, np.datetime64) and init_value == epoch:
            errors.append("init_date is 1970-01-01T00:00:00 (epoch placeholder)")
        if np.issubdtype(np.asarray(init_value).dtype, np.datetime64):
            expected_valid = init_value + np.timedelta64(step_value, "h")
            if valid_value != expected_valid:
                errors.append(f"valid_time mismatch: expected {expected_valid!s}, found {valid_value!s}")

    return errors
