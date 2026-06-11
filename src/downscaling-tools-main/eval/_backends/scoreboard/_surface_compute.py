"""Surface-loss computation kernel.

Extracted from eval/jobs/scoreboard_surface_loss.py to keep scoreboard modules
independent from eval.jobs CLI shims.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Surface variables and weights from anemoi-config/training/scalers/downscaling.yaml.
SURFACE_VARIABLES: OrderedDict[str, float] = OrderedDict(
    [
        ("10u", 2.5),
        ("10v", 2.5),
        ("2d", 2.0),
        ("2t", 2.0),
        ("msl", 2.0),
        ("skt", 0.5),
        ("sp", 1.5),
        ("tcw", 1.0),
    ]
)
TOTAL_WEIGHT = float(sum(SURFACE_VARIABLES.values()))  # 14.0
SURFACE_NORMALIZATION_SCHEME = "truth-std"


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        raise ValueError("Cannot normalize empty area weights")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Area weights contain non-finite values")
    if np.any(weights < 0):
        raise ValueError("Area weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Area weights sum must be > 0")
    return weights / total


def _decode_weather_state(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _infer_spatial_dims(ds: xr.Dataset, da: xr.DataArray) -> tuple[str, ...]:
    if "grid_point_hres" in da.dims:
        return ("grid_point_hres",)

    lon_dims = tuple(ds["lon_hres"].dims)
    lat_dims = tuple(ds["lat_hres"].dims)
    if lon_dims == lat_dims and lon_dims and all(dim in da.dims for dim in lon_dims):
        return lon_dims

    if ds["lon_hres"].ndim == 1 and ds["lat_hres"].ndim == 1:
        candidate = tuple(dict.fromkeys((*lat_dims, *lon_dims)))
        if candidate and all(dim in da.dims for dim in candidate):
            return candidate

    raise ValueError(f"Could not infer spatial dimensions from {da.dims}")


def _to_member_point_weather(da: xr.DataArray, ds: xr.Dataset, *, label: str) -> xr.DataArray:
    if "weather_state" not in da.dims:
        raise ValueError(f"{label} is missing 'weather_state' dimension: {da.dims}")

    if "sample" in da.dims and da.sizes["sample"] == 1:
        da = da.isel(sample=0, drop=True)

    spatial_dims = _infer_spatial_dims(ds, da)
    member_dims = [dim for dim in da.dims if dim not in (*spatial_dims, "weather_state")]
    if not member_dims:
        da = da.expand_dims({"member": [0]})
        member_dim = "member"
    elif len(member_dims) == 1:
        member_dim = member_dims[0]
    else:
        da = da.stack(member=member_dims)
        member_dim = "member"

    if spatial_dims != ("grid_point_hres",):
        da = da.stack(grid_point_hres=spatial_dims)
    if member_dim != "member":
        da = da.rename({member_dim: "member"})

    return da.transpose("member", "grid_point_hres", "weather_state")


def _area_weights(ds: xr.Dataset, n_points: int) -> np.ndarray:
    if "area_weight" in ds.variables:
        candidate = np.asarray(ds["area_weight"].values, dtype=np.float64).reshape(-1)
        if candidate.size == n_points:
            return _normalize_weights(candidate)

    if "lat_hres" in ds.variables:
        lat_da = ds["lat_hres"]
        lat_vals = np.asarray(lat_da.values, dtype=np.float64).reshape(-1)
        if lat_vals.size == n_points:
            return _normalize_weights(np.cos(np.deg2rad(lat_vals)))
        if lat_da.ndim == 1 and "lon_hres" in ds.variables and ds["lon_hres"].ndim == 1:
            lat_grid, _ = xr.broadcast(lat_da, ds["lon_hres"])
            lat_grid_vals = np.asarray(lat_grid.values, dtype=np.float64).reshape(-1)
            if lat_grid_vals.size == n_points:
                return _normalize_weights(np.cos(np.deg2rad(lat_grid_vals)))

    print("Warning: could not infer area weights, using uniform weights")
    return np.full(n_points, 1.0 / n_points, dtype=np.float64)


def _weather_state_index(ds: xr.Dataset) -> dict[str, int]:
    if "weather_state" not in ds.variables and "weather_state" not in ds.coords:
        raise ValueError("Dataset is missing weather_state coordinate")
    weather_states = [_decode_weather_state(v) for v in ds["weather_state"].values.tolist()]
    return {name: idx for idx, name in enumerate(weather_states)}


def _member_area_weighted_mse(
    y_pred_member_point: np.ndarray,
    y_member_point: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    sq_error = np.square(y_pred_member_point - y_member_point)
    valid = np.isfinite(sq_error)

    weighted_valid = valid * weights[np.newaxis, :]
    weight_sums = weighted_valid.sum(axis=1)
    if np.any(weight_sums <= 0.0):
        raise ValueError("No valid weighted points available for one or more members")

    numerators = np.where(valid, sq_error, 0.0) * weights[np.newaxis, :]
    return numerators.sum(axis=1) / weight_sums


def _update_running_moments(state: dict[str, float], values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    count = int(finite.size)
    mean = float(np.mean(finite))
    m2 = float(np.sum(np.square(finite - mean)))
    previous_count = int(state["count"])
    if previous_count == 0:
        state["count"] = count
        state["mean"] = mean
        state["m2"] = m2
        return
    delta = mean - float(state["mean"])
    total = previous_count + count
    state["m2"] = float(state["m2"]) + m2 + delta * delta * previous_count * count / total
    state["mean"] = (float(state["mean"]) * previous_count + mean * count) / total
    state["count"] = total


def _variance_from_running_moments(state: dict[str, float]) -> float | None:
    count = int(state["count"])
    if count <= 0:
        return None
    variance = float(state["m2"]) / count
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    return variance


def process_predictions_dir(
    predictions_dir: Path,
    *,
    prediction_var: str = "y_pred",
    truth_var: str = "y",
) -> dict[str, Any]:
    pred_files = sorted(predictions_dir.glob("predictions_*.nc"))
    if not pred_files:
        raise ValueError(f"No prediction files found in {predictions_dir}")

    print(f"Found {len(pred_files)} prediction files")
    active_surface_variables: OrderedDict[str, float] | None = None
    missing_surface_variables: list[str] = []
    var_mse_values: dict[str, list[float]] = {}
    truth_moments: dict[str, dict[str, float]] = {}

    for pred_file in pred_files:
        print(f"Processing {pred_file.name}...")
        with xr.open_dataset(pred_file, cache=False) as ds:
            for required_var in (prediction_var, truth_var, "weather_state"):
                if required_var not in ds:
                    raise ValueError(f"{pred_file}: missing required variable '{required_var}'")

            y_pred = _to_member_point_weather(ds[prediction_var], ds, label=prediction_var)
            y_true = _to_member_point_weather(ds[truth_var], ds, label=truth_var)

            if y_pred.sizes["grid_point_hres"] != y_true.sizes["grid_point_hres"]:
                raise ValueError(
                    f"{pred_file}: mismatched grid_point_hres sizes "
                    f"({y_pred.sizes['grid_point_hres']} vs {y_true.sizes['grid_point_hres']})"
                )
            if y_pred.sizes["weather_state"] != y_true.sizes["weather_state"]:
                raise ValueError(
                    f"{pred_file}: mismatched weather_state sizes "
                    f"({y_pred.sizes['weather_state']} vs {y_true.sizes['weather_state']})"
                )

            if y_true.sizes["member"] == 1 and y_pred.sizes["member"] > 1:
                y_true = y_true.isel(member=0, drop=True).expand_dims(member=y_pred["member"].values)
            elif y_true.sizes["member"] != y_pred.sizes["member"]:
                raise ValueError(
                    f"{pred_file}: mismatched member sizes "
                    f"({y_pred.sizes['member']} vs {y_true.sizes['member']})"
                )

            n_points = int(y_pred.sizes["grid_point_hres"])
            weights = _area_weights(ds, n_points)
            ws_index = _weather_state_index(ds)
            file_surface_variables = OrderedDict(
                (var, weight) for var, weight in SURFACE_VARIABLES.items() if var in ws_index
            )
            file_missing_variables = [var for var in SURFACE_VARIABLES if var not in ws_index]
            if not file_surface_variables:
                raise ValueError(
                    f"{pred_file}: none of the supported surface weather_state entries were found; "
                    f"available entries={sorted(ws_index)}"
                )
            if active_surface_variables is None:
                active_surface_variables = file_surface_variables
                missing_surface_variables = file_missing_variables
                var_mse_values = {var: [] for var in active_surface_variables}
                truth_moments = {
                    var: {"count": 0.0, "mean": 0.0, "m2": 0.0}
                    for var in active_surface_variables
                }
                if missing_surface_variables:
                    print(
                        "Warning: proceeding without missing surface weather_state entries: "
                        + ", ".join(missing_surface_variables)
                    )
            elif tuple(file_surface_variables) != tuple(active_surface_variables):
                raise ValueError(
                    f"{pred_file}: inconsistent surface weather_state coverage; "
                    f"expected {list(active_surface_variables)}, got {list(file_surface_variables)}"
                )

            for var_name in active_surface_variables:
                idx = ws_index[var_name]
                # Load one surface field at a time to keep memory bounded on large O320 files.
                y_pred_vals = np.asarray(y_pred.isel(weather_state=idx).values, dtype=np.float64)
                y_true_vals = np.asarray(y_true.isel(weather_state=idx).values, dtype=np.float64)
                per_member_mse = _member_area_weighted_mse(
                    y_pred_vals,
                    y_true_vals,
                    weights,
                )
                var_mse_values[var_name].extend(per_member_mse.tolist())
                _update_running_moments(truth_moments[var_name], y_true_vals)

    if active_surface_variables is None:
        raise ValueError(f"No supported surface weather_state entries found under {predictions_dir}")

    per_var_results: dict[str, dict[str, Any]] = {}
    weighted_surface_mse = 0.0
    weighted_surface_nmse = 0.0
    has_weighted_surface_nmse = False
    n_member_samples_per_variable: int | None = None
    total_weight = float(sum(active_surface_variables.values()))

    for var_name, var_weight in active_surface_variables.items():
        values = np.asarray(var_mse_values[var_name], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No MSE values collected for required variable '{var_name}'")
        mean_mse = float(np.mean(values))
        n_samples = int(values.size)
        normalized_weight = float(var_weight / total_weight)
        truth_variance = _variance_from_running_moments(truth_moments[var_name])
        truth_std = float(np.sqrt(truth_variance)) if truth_variance is not None else None
        mean_nmse = mean_mse / truth_variance if truth_variance is not None else None

        per_var_results[var_name] = {
            "mean_mse": mean_mse,
            "weight": float(var_weight),
            "normalized_weight": normalized_weight,
            "n_member_samples": n_samples,
        }
        if truth_std is not None:
            per_var_results[var_name]["truth_std"] = truth_std
        if mean_nmse is not None and np.isfinite(mean_nmse):
            per_var_results[var_name]["mean_nmse"] = float(mean_nmse)
        weighted_surface_mse += mean_mse * normalized_weight
        if mean_nmse is not None and np.isfinite(mean_nmse):
            weighted_surface_nmse += float(mean_nmse) * normalized_weight
            has_weighted_surface_nmse = True

        if n_member_samples_per_variable is None:
            n_member_samples_per_variable = n_samples
        elif n_member_samples_per_variable != n_samples:
            raise ValueError(
                "Inconsistent member sample counts across variables; cannot build aggregate summary"
            )

    return {
        "weighted_surface_mse": float(weighted_surface_mse),
        "weighted_surface_nmse": float(weighted_surface_nmse) if has_weighted_surface_nmse else None,
        "normalization_scheme": SURFACE_NORMALIZATION_SCHEME,
        "prediction_var": prediction_var,
        "truth_var": truth_var,
        "total_weight": total_weight,
        "requested_surface_variables": list(SURFACE_VARIABLES.keys()),
        "available_surface_variables": list(active_surface_variables.keys()),
        "missing_surface_variables": missing_surface_variables,
        "total_requested_weight": TOTAL_WEIGHT,
        "n_prediction_files": len(pred_files),
        "n_member_samples_per_variable": int(n_member_samples_per_variable or 0),
        "variables": per_var_results,
    }

