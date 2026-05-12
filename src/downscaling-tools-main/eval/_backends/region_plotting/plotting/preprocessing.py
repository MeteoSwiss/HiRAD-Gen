"""Preprocessing helpers shared by region plotting scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from eval.checkpoint_interpolation import CheckpointResidualInterpolator, resolve_checkpoint_path


def ensure_member_zero_plot_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add member-zero aliases used by legacy plotting code when they are missing."""
    alias_specs = (
        ("x", "x_0"),
        ("x_interp", "x_interp_0"),
        ("y", "y_0"),
        ("y_pred", "y_pred_0"),
    )
    updates: dict[str, xr.DataArray] = {}
    for base_name, alias_name in alias_specs:
        if alias_name in ds.variables or base_name not in ds.variables:
            continue
        da = ds[base_name]
        alias = da.isel(ensemble_member=0) if "ensemble_member" in da.dims else da
        alias = alias.rename(alias_name)
        alias.attrs = dict(da.attrs)
        updates[alias_name] = alias
    if updates:
        ds = ds.assign(**updates)
    return ds


def ensure_x_interp_for_plotting(
    ds: xr.Dataset,
    *,
    predictions_path: str | Path | None = None,
    checkpoint_path: str = "",
) -> xr.Dataset:
    """Ensure ``x_interp`` and member-zero aliases exist for plotting."""
    if "x_interp" not in ds.variables:
        if "x" not in ds.variables:
            return ensure_member_zero_plot_variables(ds)

        pred_dir = Path(predictions_path).expanduser().resolve().parent if predictions_path else Path.cwd()
        resolved_checkpoint = resolve_checkpoint_path(pred_dir=pred_dir, ds=ds, explicit_path=checkpoint_path)
        if resolved_checkpoint is None:
            return ensure_member_zero_plot_variables(ds)

        interpolator = CheckpointResidualInterpolator(resolved_checkpoint)
        interpolated = interpolator.interpolate(np.asarray(ds["x"].values))
        x_interp = xr.DataArray(
            interpolated.astype(np.float32),
            dims=ds["y_pred"].dims,
            coords={dim: ds.coords[dim] for dim in ds["y_pred"].dims if dim in ds.coords},
            attrs=dict(ds["y_pred"].attrs),
            name="x_interp",
        )
        if "lon" not in x_interp.attrs and "lon_hres" in ds.coords:
            x_interp.attrs["lon"] = "lon_hres"
        if "lat" not in x_interp.attrs and "lat_hres" in ds.coords:
            x_interp.attrs["lat"] = "lat_hres"
        ds = ds.assign(x_interp=x_interp)
    return ensure_member_zero_plot_variables(ds)
