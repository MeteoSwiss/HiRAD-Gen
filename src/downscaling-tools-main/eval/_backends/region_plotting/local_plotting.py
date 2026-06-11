from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.training.diagnostics.maps import Coastlines
from matplotlib.backends.backend_pdf import PdfPages

from eval.checkpoint_interpolation import CheckpointResidualInterpolator, resolve_checkpoint_path
from .plotting.coordinate_utils import (
    _coord_name_for_array as shared_coord_name_for_array,
    get_region_ds as shared_get_region_ds,
    legacy_plotter_regions_for_grid,
)
from .plotting.datetime_utils import extract_date_from_dataset
from .plotting.preprocessing import ensure_member_zero_plot_variables as shared_ensure_member_zero_plot_variables
from .plotting.variable_utils import (
    DERIVED_MODEL_VARIABLE_SPECS as SHARED_DERIVED_MODEL_VARIABLE_SPECS,
    get_plot_data_array as shared_get_plot_data_array,
    is_residual_plot_variable as shared_is_residual_plot_variable,
    supports_plot_variable as shared_supports_plot_variable,
)

continents = Coastlines()
LOG = logging.getLogger(__name__)

DERIVED_MODEL_VARIABLE_SPECS = SHARED_DERIVED_MODEL_VARIABLE_SPECS


def get_minmax_weather_states(
    ds: xr.Dataset, weather_states: list[str], list_model_variables: list[str]
) -> dict[str, list[float]]:
    minmax_weather_states: dict[str, list[float]] = {}
    for weather_state in weather_states:
        fields: list[np.ndarray] = []
        for model_var in list_model_variables:
            if not supports_plot_variable(ds, model_var):
                continue
            da = get_plot_data_array(ds, model_var)
            if "weather_state" in da.dims:
                da = da.sel(weather_state=weather_state)
            fields.append(np.asarray(da.values).reshape(-1))
        if not fields:
            continue
        fields_val = np.concatenate(fields)
        minmax_weather_states[weather_state] = [
            float(np.nanmin(fields_val)),
            float(np.nanmax(fields_val)),
        ]
    return minmax_weather_states


def supports_plot_variable(ds: xr.Dataset, model_var: str) -> bool:
    return shared_supports_plot_variable(ds, model_var)


def _coord_name_for_array(ds: xr.Dataset, da: xr.DataArray, axis: str) -> str:
    return shared_coord_name_for_array(ds, da, axis)


def get_plot_data_array(ds: xr.Dataset, model_var: str) -> xr.DataArray:
    return shared_get_plot_data_array(ds, model_var)


def ensure_x_interp_for_plotting(
    ds: xr.Dataset,
    *,
    predictions_path: str | Path | None = None,
    checkpoint_path: str = "",
) -> xr.Dataset:
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


def ensure_member_zero_plot_variables(ds: xr.Dataset) -> xr.Dataset:
    return shared_ensure_member_zero_plot_variables(ds)


def plot_variable_title(model_var: str) -> str:
    return DERIVED_MODEL_VARIABLE_SPECS.get(model_var, {}).get("title", model_var)


def is_residual_plot_variable(model_var: str) -> bool:
    return shared_is_residual_plot_variable(model_var)


def _residual_vmax(da: xr.DataArray) -> float:
    values = np.asarray(da.values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(finite))
    return vmax if vmax > 0 else 1.0


def plot_x_y(
    ds_sample: xr.Dataset,
    list_model_variables: list[str],
    weather_states: list[str],
    consistent_cbar: list[str] = [
        "x_0",
        "x_interp_0",
        "y_0",
        "y_pred_0",
        "x",
        "x_interp",
        "y",
        "y_pred",
        "x_interp_0",
        "y_pred_0",
        "y_pred_1",
        "y_pred_2",
        "x_interp_1",
        "x_interp_2",
        "x_0",
        "x_1",
        "x_2",
        "y_0",
        "y_1",
        "y_2",
    ],
    title: str | None = None,
):
    list_model_variables = [v for v in list_model_variables if supports_plot_variable(ds_sample, v)]
    overlap = [
        model_var
        for model_var in list_model_variables
        if model_var in consistent_cbar and supports_plot_variable(ds_sample, model_var)
    ]
    minmax_weather_states = get_minmax_weather_states(ds_sample, weather_states, overlap)

    figsize = (len(list_model_variables) * 4, len(weather_states) * 3)
    fig, axs = plt.subplots(len(weather_states), len(list_model_variables), figsize=figsize)

    if len(list_model_variables) == 1:
        axs = np.array([axs]).transpose()
    if len(weather_states) == 1:
        axs = np.array([axs])

    ims = {}
    cbars = {}
    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):
            da = get_plot_data_array(ds_sample, model_var)
            lon_name = _coord_name_for_array(ds_sample, da, "lon")
            lat_name = _coord_name_for_array(ds_sample, da, "lat")
            if "weather_state" in da.dims:
                da = da.sel(weather_state=weather_state)
            scatter_params = dict(
                x=ds_sample[lon_name].values,
                y=ds_sample[lat_name].values,
                c=da.values,
                s=75_000 / len(ds_sample[lon_name].values),
                alpha=1.0,
                rasterized=True,
            )

            if model_var in consistent_cbar and weather_state in minmax_weather_states:
                scatter_params.update(
                    vmin=minmax_weather_states[weather_state][0],
                    vmax=minmax_weather_states[weather_state][1],
                    cmap="viridis",
                )
            elif is_residual_plot_variable(model_var):
                vmax = _residual_vmax(da)
                scatter_params.update(vmin=-vmax, vmax=vmax, cmap="bwr")
            else:
                vmax = float(np.nanmax(da.values))
                vmin = float(np.nanmin(da.values))
                scatter_params.update(vmin=vmin, vmax=vmax, cmap="viridis")

            ims[(i_ax0, i_ax1)] = axs[i_ax0, i_ax1].scatter(**scatter_params)
            cbars[(i_ax0, i_ax1)] = plt.colorbar(
                ims[(i_ax0, i_ax1)],
                ax=axs[i_ax0, i_ax1],
                orientation="vertical",
                pad=0.05,
            )

    for i_ax0, _weather_state in enumerate(weather_states):
        axs[i_ax0, 0].set_ylabel("Latitude (°)", fontsize=12)
    for i_ax1, _model_var in enumerate(list_model_variables):
        axs[-1, i_ax1].set_xlabel("Longitude (°)", fontsize=12)

    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):
            axs[i_ax0, i_ax1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            axs[i_ax0, i_ax1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            axs[i_ax0, i_ax1].tick_params(axis="both", which="major", labelsize=10)
            axs[i_ax0, i_ax1].set_title(f"{plot_variable_title(model_var)} - {weather_state}")

            if "region" in ds_sample.attrs:
                axs[i_ax0, i_ax1].set_xlim(ds_sample.attrs["region"][2], ds_sample.attrs["region"][3])
                axs[i_ax0, i_ax1].set_ylim(ds_sample.attrs["region"][0], ds_sample.attrs["region"][1])
            continents.plot_continents(axs[i_ax0, i_ax1])
            axs[i_ax0, i_ax1].set_aspect("auto", adjustable=None)
            axs[i_ax0, i_ax1].grid(False)
            axs[i_ax0, i_ax1].patch.set_edgecolor("black")
            axs[i_ax0, i_ax1].patch.set_linewidth(2)
            cbars[(i_ax0, i_ax1)].outline.set_edgecolor("black")
            cbars[(i_ax0, i_ax1)].outline.set_linewidth(1.0)
            cbars[(i_ax0, i_ax1)].ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, fontsize=16, y=1.0)
    else:
        fig.suptitle(extract_date_from_dataset(ds_sample) or "Unknown date", fontsize=16, y=1.0)
    fig.tight_layout()
    return fig


def get_region_ds(ds: xr.Dataset, region_box: Union[str, list[int]] = "default") -> xr.Dataset:
    return shared_get_region_ds(ds, region_box)


@dataclass
class LocalInferencePlotter:
    """Deprecated: use ``render_region_suite_from_predictions_file()`` instead."""

    dir_exp: str
    name_exp: str
    name_predictions_file: str

    def __post_init__(self):
        warnings.warn(
            "LocalInferencePlotter is deprecated. Use render_region_suite_from_predictions_file() from plot_regions.py.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ds = xr.open_dataset(os.path.join(self.dir_exp, self.name_exp, self.name_predictions_file))
        self.ds = ensure_x_interp_for_plotting(
            self.ds,
            predictions_path=Path(self.dir_exp) / self.name_exp / self.name_predictions_file,
        )
        self.regions = legacy_plotter_regions_for_grid(str(self.ds.attrs["grid"]))

    def save_plot(
        self,
        list_regions: list[str],
        list_model_variables: list[str] = ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"],
        weather_states: list[str] = ["10u", "10v", "2t", "msl", "tp", "z_500", "u_850", "v_850", "t_850"],
        num_samples_to_plot: int = 2,
    ) -> None:
        selected_model_variables = [v for v in list_model_variables if supports_plot_variable(self.ds, v)]
        if not selected_model_variables:
            raise ValueError(
                f"None of the requested model variables are available in {self.name_predictions_file}. "
                f"Requested={list_model_variables}"
            )
        available_weather_states = [str(v) for v in self.ds["weather_state"].values.tolist()]
        selected_weather_states = [w for w in weather_states if w in available_weather_states]
        if not selected_weather_states:
            selected_weather_states = available_weather_states

        pdf_path = f"{self.dir_exp}/{self.name_exp}/all_regions_plots.pdf"
        if os.path.exists(pdf_path):
            LOG.info("Removing existing PDF at %s", pdf_path)
            os.remove(pdf_path)
        with PdfPages(pdf_path) as pdf:
            for region in list_regions:
                LOG.info("Plotting region %s", region)
                region_ds = get_region_ds(self.ds, region)
                region_ds.attrs["region_name"] = region

                if "sample" in region_ds.dims:
                    n_available = int(region_ds.sizes.get("sample", 0))
                    n_to_plot = min(num_samples_to_plot, n_available)
                    for sample in range(n_to_plot):
                        fig = plot_x_y(
                            ds_sample=region_ds.sel(sample=sample),
                            list_model_variables=selected_model_variables,
                            weather_states=selected_weather_states,
                            title=f"{region} - sample {sample}",
                        )
                        pdf.savefig(fig)
                        plt.close(fig)
                else:
                    sample_count = 0
                    for step in region_ds.step.values:
                        for ft in np.atleast_1d(region_ds.forecast_reference_time.values):
                            if sample_count >= num_samples_to_plot:
                                break
                            fig = plot_x_y(
                                ds_sample=region_ds.sel(step=step, forecast_reference_time=ft),
                                list_model_variables=selected_model_variables,
                                weather_states=selected_weather_states,
                                title=f"{region} - step {step} - forecast {pd.to_datetime(ft).strftime('%Y-%m-%d')}",
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                            sample_count += 1
                        if sample_count >= num_samples_to_plot:
                            break

        LOG.info("Plot saved successfully at %s", pdf_path)
