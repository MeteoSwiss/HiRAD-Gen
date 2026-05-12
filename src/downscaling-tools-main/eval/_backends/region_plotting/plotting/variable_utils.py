"""Variable selection helpers shared by region plotting scripts."""
from __future__ import annotations

import xarray as xr

DERIVED_MODEL_VARIABLE_SPECS: dict[str, dict[str, str]] = {
    "residuals_pred_0": {
        "left": "x_interp_0",
        "right": "y_pred_0",
        "title": "residuals_pred_0",
    },
    "residuals_pred": {
        "left": "x_interp",
        "right": "y_pred",
        "title": "residuals_pred",
    },
    "x_interp_minus_y_pred": {
        "left": "x_interp",
        "right": "y_pred",
        "title": "residuals_pred",
    },
    "residuals_0": {
        "left": "x_interp_0",
        "right": "y_0",
        "title": "residuals_0",
    },
    "residuals": {
        "left": "x_interp",
        "right": "y",
        "title": "residuals",
    },
    "x_interp_minus_y": {
        "left": "x_interp",
        "right": "y",
        "title": "residuals",
    },
}


def supports_plot_variable(ds: xr.Dataset, model_var: str) -> bool:
    """Return True when *model_var* is directly present or derivable."""
    if model_var in ds.variables:
        return True
    spec = DERIVED_MODEL_VARIABLE_SPECS.get(model_var)
    if spec is None:
        return False
    return spec["left"] in ds.variables and spec["right"] in ds.variables


def get_plot_data_array(ds: xr.Dataset, model_var: str) -> xr.DataArray:
    """Return the plottable data array for *model_var*, deriving residuals when needed."""
    if model_var in ds.variables:
        return ds[model_var]

    spec = DERIVED_MODEL_VARIABLE_SPECS.get(model_var)
    if spec is None:
        raise KeyError(f"Unsupported model variable: {model_var}")

    derived = (ds[spec["left"]] - ds[spec["right"]]).rename(model_var)
    attrs = dict(ds[spec["left"]].attrs)
    if "lon" not in attrs and "lon" in ds[spec["right"]].attrs:
        attrs["lon"] = ds[spec["right"]].attrs["lon"]
    if "lat" not in attrs and "lat" in ds[spec["right"]].attrs:
        attrs["lat"] = ds[spec["right"]].attrs["lat"]
    derived.attrs = attrs
    return derived


def is_residual_plot_variable(model_var: str) -> bool:
    """Return True for residual-style plotting variables."""
    return model_var == "y_diff" or model_var in DERIVED_MODEL_VARIABLE_SPECS
