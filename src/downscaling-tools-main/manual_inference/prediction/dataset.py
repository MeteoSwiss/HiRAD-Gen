from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr


OUTPUT_WEATHER_STATE_MODE_CHOICES = ("all", "surface-plus-core-pl")
_SURFACE_PLUS_CORE_PL_WEATHER_STATES = frozenset({"t_850", "z_500"})


def infer_grid_attr(*, lres_len: int | None, hres_len: int | None) -> str | None:
    if hres_len in (421120,):
        return "O320"
    if hres_len in (6599680,):
        return "O1280"
    if lres_len in (40320,):
        return "O320"
    if lres_len in (421120,):
        return "O1280"
    return None


def ensure_sample_dim(ds: xr.Dataset) -> xr.Dataset:
    if "sample" not in ds.dims:
        ds = ds.expand_dims("sample")
    return ds


def add_member_views(ds: xr.Dataset) -> xr.Dataset:
    if "x" in ds:
        if "ensemble_member" in ds["x"].dims:
            for i in range(ds["x"].sizes["ensemble_member"]):
                name = f"x_{i}"
                if name in ds:
                    continue
                ds[name] = ds["x"].isel(ensemble_member=i)
                ds[name].attrs.update(ds["x"].attrs)
        elif "x_0" not in ds:
            ds["x_0"] = ds["x"]
            ds["x_0"].attrs.update(ds["x"].attrs)
    if "y" in ds:
        if "ensemble_member" in ds["y"].dims:
            for i in range(ds["y"].sizes["ensemble_member"]):
                name = f"y_{i}"
                if name in ds:
                    continue
                ds[name] = ds["y"].isel(ensemble_member=i)
                ds[name].attrs.update(ds["y"].attrs)
        elif "y_0" not in ds:
            ds["y_0"] = ds["y"]
            ds["y_0"].attrs.update(ds["y"].attrs)
    if "x_interp" in ds and "ensemble_member" in ds["x_interp"].dims:
        for i in range(ds["x_interp"].sizes["ensemble_member"]):
            name = f"x_interp_{i}"
            if name in ds:
                continue
            ds[name] = ds["x_interp"].isel(ensemble_member=i)
            ds[name].attrs.update(ds["x_interp"].attrs)
    if "y_pred" in ds and "ensemble_member" in ds["y_pred"].dims:
        for i in range(ds["y_pred"].sizes["ensemble_member"]):
            name = f"y_pred_{i}"
            if name in ds:
                continue
            ds[name] = ds["y_pred"].isel(ensemble_member=i)
            ds[name].attrs.update(ds["y_pred"].attrs)
    return ds


def resolve_output_weather_states(
    *,
    weather_states: Sequence[str],
    mode: str = "all",
    explicit_weather_states: Sequence[str] | None = None,
) -> tuple[list[str], list[int]]:
    states = [str(name) for name in weather_states]

    if explicit_weather_states is not None:
        requested = [str(name).strip() for name in explicit_weather_states if str(name).strip()]
        if not requested:
            raise ValueError("Explicit output weather states cannot be empty.")
        missing = [name for name in requested if name not in states]
        if missing:
            raise ValueError(
                "Requested output weather state(s) not found: " + ", ".join(missing)
            )
        return requested, [states.index(name) for name in requested]

    if mode == "all":
        return states, list(range(len(states)))

    if mode == "surface-plus-core-pl":
        selected = [
            name
            for name in states
            if "_" not in name or name in _SURFACE_PLUS_CORE_PL_WEATHER_STATES
        ]
        return selected, [states.index(name) for name in selected]

    raise ValueError(
        f"Unsupported output weather state mode {mode!r}. "
        f"Expected one of {OUTPUT_WEATHER_STATE_MODE_CHOICES}."
    )


def build_predictions_dataset(
    *,
    x: np.ndarray,
    y: np.ndarray | None,
    y_pred: np.ndarray,
    lon_lres: np.ndarray,
    lat_lres: np.ndarray,
    lon_hres: np.ndarray,
    lat_hres: np.ndarray,
    weather_states: Sequence[str],
    dates: Sequence | None,
    member_ids: Sequence[int],
    x_interp: np.ndarray | None = None,
    include_member_views: bool = True,
) -> xr.Dataset:
    dates = np.asarray(dates) if dates is not None else np.arange(x.shape[0])
    if x.ndim == 3:
        x_dims = ["sample", "grid_point_lres", "weather_state"]
    elif x.ndim == 4:
        x_dims = ["sample", "ensemble_member", "grid_point_lres", "weather_state"]
        if x.shape[1] != len(member_ids):
            raise ValueError(
                f"x ensemble dimension ({x.shape[1]}) does not match member_ids length ({len(member_ids)})."
            )
    else:
        raise ValueError(f"Unsupported x shape {x.shape}. Expected 3D or 4D array.")

    ds_vars = {
        "x": (x_dims, x),
        "y_pred": (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            y_pred,
        ),
        "date": (["sample"], dates),
        "lon_lres": (["grid_point_lres"], lon_lres),
        "lat_lres": (["grid_point_lres"], lat_lres),
        "lon_hres": (["grid_point_hres"], lon_hres),
        "lat_hres": (["grid_point_hres"], lat_hres),
    }
    if x_interp is not None:
        if x_interp.ndim != 4:
            raise ValueError(f"Unsupported x_interp shape {x_interp.shape}. Expected 4D array.")
        if x_interp.shape[0] != x.shape[0]:
            raise ValueError(
                f"x_interp sample dimension ({x_interp.shape[0]}) does not match x sample dimension ({x.shape[0]})."
            )
        if x_interp.shape[1] != len(member_ids):
            raise ValueError(
                f"x_interp ensemble dimension ({x_interp.shape[1]}) does not match member_ids length ({len(member_ids)})."
            )
        if x_interp.shape[2] != lon_hres.shape[0]:
            raise ValueError(
                f"x_interp hres grid dimension ({x_interp.shape[2]}) does not match lon_hres length ({lon_hres.shape[0]})."
            )
        if x_interp.shape[3] != len(weather_states):
            raise ValueError(
                f"x_interp weather-state dimension ({x_interp.shape[3]}) does not match weather_states length ({len(weather_states)})."
            )
        ds_vars["x_interp"] = (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            x_interp,
        )
    if y is not None:
        if y.ndim == 3:
            y_dims = ["sample", "grid_point_hres", "weather_state"]
        elif y.ndim == 4:
            y_dims = ["sample", "ensemble_member", "grid_point_hres", "weather_state"]
            if y.shape[1] != len(member_ids):
                raise ValueError(
                    f"y ensemble dimension ({y.shape[1]}) does not match member_ids length ({len(member_ids)})."
                )
        else:
            raise ValueError(f"Unsupported y shape {y.shape}. Expected 3D or 4D array.")
        ds_vars["y"] = (y_dims, y)

    ds = xr.Dataset(
        ds_vars,
        coords={
            "sample": range(x.shape[0]),
            "ensemble_member": list(member_ids),
            "grid_point_lres": range(lon_lres.shape[0]),
            "grid_point_hres": range(lon_hres.shape[0]),
            "weather_state": list(weather_states),
        },
    )

    ds["x"].attrs["lon"] = "lon_lres"
    ds["x"].attrs["lat"] = "lat_lres"
    ds["y_pred"].attrs["lon"] = "lon_hres"
    ds["y_pred"].attrs["lat"] = "lat_hres"
    if "x_interp" in ds:
        ds["x_interp"].attrs["lon"] = "lon_hres"
        ds["x_interp"].attrs["lat"] = "lat_hres"
        ds["x_interp"].attrs["long_name"] = "low-resolution inputs interpolated to the high-resolution output grid"
    if "y" in ds:
        ds["y"].attrs["lon"] = "lon_hres"
        ds["y"].attrs["lat"] = "lat_hres"

    ds["lon_hres"] = ((ds.lon_hres + 180) % 360) - 180
    ds["lon_lres"] = ((ds.lon_lres + 180) % 360) - 180

    grid_attr = infer_grid_attr(
        lres_len=int(ds.sizes.get("grid_point_lres", 0)),
        hres_len=int(ds.sizes.get("grid_point_hres", 0)),
    )
    if grid_attr:
        ds.attrs["grid"] = grid_attr

    ds = ensure_sample_dim(ds)
    if include_member_views:
        ds = add_member_views(ds)
    return ds
