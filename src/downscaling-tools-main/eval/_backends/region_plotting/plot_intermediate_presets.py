from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from anemoi.training.diagnostics.maps import Coastlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.tri import LinearTriInterpolator, Triangulation

from .local_plotting import plot_x_y
from .plotting.config import RENDER_DPI
from .plotting.coordinate_utils import get_region_ds
from .plotting.datetime_utils import is_valid_date


CONTINENTS = Coastlines()
DEFAULT_WEATHER = ["10u", "10v", "2t", "msl"]


def _sigmas_from_cfg(cfg_json: str, fallback_num_steps: int = 20) -> np.ndarray:
    cfg = json.loads(cfg_json) if cfg_json else {}
    num_steps = int(cfg.get("num_steps", fallback_num_steps))
    sigma_max = float(cfg.get("sigma_max", 1000.0))
    sigma_min = float(cfg.get("sigma_min", 0.03))
    rho = float(cfg.get("rho", 7.0))
    ramp = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sig_no0 = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return np.concatenate([sig_no0, np.array([0.0], dtype=np.float64)])


def _build_panel_ds(
    ds: xr.Dataset,
    weather_states: list[str],
    ordered_steps: list[int],
    include_sigma: bool,
) -> tuple[xr.Dataset, list[str]]:
    ds_m = ds.isel(ensemble_member=0)
    steps = [int(s) for s in ds_m["sampling_step"].values.tolist()]
    sigmas = _sigmas_from_cfg(ds.attrs.get("sampling_config_json", ""), fallback_num_steps=max(steps) + 1)

    _y = ds_m["y"].isel(sample=0)
    if "lon" not in _y.attrs:
        _y.attrs.update({"lon": "lon_hres", "lat": "lat_hres"})
    out = xr.Dataset(
        {
            "x": ds_m["x"].isel(sample=0),
            "y": _y,
            "y_pred": ds_m["y_pred"].isel(sample=0),
            "lon_lres": ds_m["lon_lres"],
            "lat_lres": ds_m["lat_lres"],
            "lon_hres": ds_m["lon_hres"],
            "lat_hres": ds_m["lat_hres"],
        },
        attrs=ds.attrs,
    )
    if "date" in ds_m:
        date_value = ds_m["date"].isel(sample=0).values
        if is_valid_date(date_value):
            out["date"] = ds_m["date"].isel(sample=0)

    model_vars = ["x", "y", "y_pred"]
    for st in ordered_steps:
        if st not in steps:
            raise ValueError(f"Requested step {st} missing. Available steps: {steps}")
        idx = steps.index(st)
        name = f"inter_step_{st}"
        if include_sigma:
            name = f"{name} (sigma={float(sigmas[st]):.3f})"
        da = ds_m["inter_state"].isel(sample=0, sampling_step=idx).rename(name).reset_coords(drop=True)
        da.attrs.update({"lon": "lon_hres", "lat": "lat_hres"})
        out[name] = da
        model_vars.append(name)

    return out.sel(weather_state=weather_states), model_vars


def _plot_minimal_pcolor_contour(
    ds_region: xr.Dataset,
    model_vars: list[str],
    weather_states: list[str],
    title: str,
) -> plt.Figure:
    def _resolve_lon_lat_names(var_name: str) -> tuple[str, str]:
        da = ds_region[var_name]
        lon_name = da.attrs.get("lon")
        lat_name = da.attrs.get("lat")
        if lon_name and lat_name:
            return lon_name, lat_name

        # Some cached intermediate datasets do not carry lon/lat attrs on x/y/y_pred.
        if "grid_point_lres" in da.dims:
            return "lon_lres", "lat_lres"
        if "grid_point_hres" in da.dims:
            return "lon_hres", "lat_hres"

        raise KeyError(
            f"Could not infer lon/lat variables for '{var_name}' from attrs={da.attrs} and dims={da.dims}."
        )

    nrows, ncols = len(weather_states), len(model_vars)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    if nrows == 1:
        axs = np.array([axs])
    if ncols == 1:
        axs = np.array([axs]).T

    shared = ["x", "y", "y_pred"]
    shared_minmax: dict[str, tuple[float, float]] = {}
    for w in weather_states:
        vals = []
        for mv in shared:
            vals.append(ds_region[mv].sel(weather_state=w).values.ravel())
        allv = np.concatenate(vals)
        shared_minmax[w] = (float(np.nanmin(allv)), float(np.nanmax(allv)))

    for i, w in enumerate(weather_states):
        for j, mv in enumerate(model_vars):
            ax = axs[i, j]
            lon_name, lat_name = _resolve_lon_lat_names(mv)
            lon = ds_region[lon_name].values
            lat = ds_region[lat_name].values
            field = ds_region[mv].sel(weather_state=w).values

            if mv in shared:
                vmin, vmax = shared_minmax[w]
            else:
                vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = vmin - 1e-6, vmax + 1e-6

            gx = np.linspace(float(np.min(lon)), float(np.max(lon)), 140)
            gy = np.linspace(float(np.min(lat)), float(np.max(lat)), 120)
            lon2, lat2 = np.meshgrid(gx, gy)
            tri = Triangulation(lon, lat)
            grid = np.asarray(LinearTriInterpolator(tri, field)(lon2, lat2))

            levels = np.linspace(vmin, vmax, 21)
            im = ax.pcolormesh(lon2, lat2, grid, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.contour(lon2, lat2, grid, levels=levels, colors="black", linewidths=0.35)
            cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.05)
            cbar.ax.tick_params(labelsize=10)

            ax.set_title(f"{mv} - {w}", fontsize=10)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            ax.tick_params(axis="both", which="major", labelsize=10)
            if "region" in ds_region.attrs:
                ax.set_xlim(ds_region.attrs["region"][2], ds_region.attrs["region"][3])
                ax.set_ylim(ds_region.attrs["region"][0], ds_region.attrs["region"][1])
            CONTINENTS.plot_continents(ax)
            ax.set_aspect("auto")
            ax.grid(False)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth(2)

    for i in range(nrows):
        axs[i, 0].set_ylabel("Latitude (°)", fontsize=12)
    for j in range(ncols):
        axs[-1, j].set_xlabel("Longitude (°)", fontsize=12)
    fig.suptitle(title, fontsize=16, y=1.0)
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description="Preset intermediate-state region plots (Amazon baseline + minimal contour variant).")
    p.add_argument("--predictions-nc", required=True)
    p.add_argument("--out", required=True, help="Output file (.pdf or .png)")
    p.add_argument("--also-png", default="", help="Optional second PNG output")
    p.add_argument("--region", default="idalia_center")
    p.add_argument("--weather-states", default="10u,10v,2t,msl")
    p.add_argument("--ordered-steps", default="16,14,13,12,11")
    p.add_argument("--style", choices=["amazon-baseline", "minimal-pcolor-contour"], default="amazon-baseline")
    p.add_argument("--include-sigma-labels", action="store_true")
    args = p.parse_args()

    weather_states = [w.strip() for w in args.weather_states.split(",") if w.strip()]
    ordered_steps = [int(s.strip()) for s in args.ordered_steps.split(",") if s.strip()]

    with xr.open_dataset(args.predictions_nc) as ds:
        ds_plot, model_vars = _build_panel_ds(
            ds=ds,
            weather_states=weather_states,
            ordered_steps=ordered_steps,
            include_sigma=bool(args.include_sigma_labels),
        )
        ds_region = get_region_ds(ds_plot, args.region)

        if args.style == "amazon-baseline":
            fig = plot_x_y(
                ds_sample=ds_region,
                list_model_variables=model_vars,
                weather_states=weather_states,
                consistent_cbar=["x", "y", "y_pred"],
                title=f"{args.region} | sample 0",
            )
        else:
            fig = _plot_minimal_pcolor_contour(
                ds_region=ds_region,
                model_vars=model_vars,
                weather_states=weather_states,
                title=f"{args.region} | sample 0",
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".pdf":
        with PdfPages(out) as pdf:
            pdf.savefig(fig)
    else:
        fig.savefig(out, dpi=RENDER_DPI)

    if args.also_png:
        fig.savefig(Path(args.also_png), dpi=RENDER_DPI)
    plt.close(fig)

    print(out)
    if args.also_png:
        print(args.also_png)


if __name__ == "__main__":
    main()
