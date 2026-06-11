from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from anemoi.training.diagnostics.maps import Coastlines
from matplotlib.backends.backend_pdf import PdfPages

from .plotting.config import PREDICTION_REGION_BOXES, RENDER_DPI
from .plotting.coordinate_utils import get_region_ds
from .plotting.manifest import write_manifest
from .plotting.metadata import sample_meta_title
from .plotting.preprocessing import ensure_x_interp_for_plotting

DEFAULT_REGIONS = ["idalia", "franklin"]
COASTLINES = Coastlines()


def _absolute_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _state_values(ds_region: xr.Dataset, variable_name: str, weather_state: str) -> np.ndarray:
    return np.asarray(ds_region[variable_name].sel(weather_state=weather_state).values, dtype=float)


def _wind10m(ds_region: xr.Dataset, variable_name: str) -> np.ndarray:
    u = _state_values(ds_region, variable_name, "10u")
    v = _state_values(ds_region, variable_name, "10v")
    return np.sqrt(u**2 + v**2)


def _msl_hpa(ds_region: xr.Dataset, variable_name: str) -> np.ndarray:
    values = _state_values(ds_region, variable_name, "msl")
    if float(np.nanmedian(values)) > 2000.0:
        values = values * 0.01
    return values


def _levels(fields: list[np.ndarray], *, n_levels: int = 21) -> np.ndarray:
    flat = np.concatenate([field.reshape(-1) for field in fields])
    vmin = float(np.nanpercentile(flat, 2))
    vmax = float(np.nanpercentile(flat, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(flat))
        vmax = float(np.nanmax(flat))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, n_levels)


def _draw_panel(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, *, levels: np.ndarray, title: str):
    contourf = ax.tricontourf(lon, lat, field, levels=levels, cmap="viridis", extend="both")
    ax.tricontour(lon, lat, field, levels=levels, colors="black", linewidths=0.45, alpha=0.7)
    COASTLINES.plot_continents(ax)
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(False)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)
    return contourf


def _write_manifest(*, out_root: Path, predictions_path: Path, region_names: list[str], sample_index: int, ensemble_member_index: int, generated: list[str]) -> Path:
    payload = {
        "suite_kind": "storm",
        "plot_style": "tc_contour",
        "predictions_file": str(predictions_path),
        "out_dir": str(out_root),
        "regions": list(region_names),
        "sample_index": int(sample_index),
        "ensemble_member_index": int(ensemble_member_index),
        "generated_files": list(generated),
        "panel_contract": {
            "rows": ["msl_hpa", "wind10m_ms"],
            "columns": ["x_interp", "y", "y_pred"],
        },
    }
    return write_manifest(out_root=out_root, payload=payload)


def render_tc_contour_suite_from_predictions_file(
    *,
    predictions_nc: str | Path,
    out_dir: str | Path,
    region_names: list[str] | None = None,
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    also_png: bool = True,
) -> list[str]:
    predictions_path = _absolute_path(predictions_nc)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    out_root = _absolute_path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_pdf = out_root / "all_regions_plots.pdf"
    generated: list[str] = [str(combined_pdf)]

    names = region_names or DEFAULT_REGIONS
    unknown = [name for name in names if name not in PREDICTION_REGION_BOXES]
    if unknown:
        raise ValueError(f"Unknown region names: {unknown}. Known regions: {sorted(PREDICTION_REGION_BOXES)}")

    with xr.open_dataset(predictions_path) as ds:
        if "sample" in ds.dims:
            ds = ds.isel(sample=sample_index)
        if "ensemble_member" in ds.dims:
            ds = ds.isel(ensemble_member=ensemble_member_index)
        ds = ensure_x_interp_for_plotting(ds, predictions_path=predictions_path)

        for required in ("x_interp", "y", "y_pred"):
            if required not in ds.variables:
                raise ValueError(f"Missing required variable for TC contour suite: {required}")
        for weather_state in ("msl", "10u", "10v"):
            if weather_state not in ds["weather_state"].values:
                raise ValueError(f"Missing required weather_state={weather_state} in {predictions_path}")

        lon = np.asarray(ds["lon_hres"].values, dtype=float)
        lat = np.asarray(ds["lat_hres"].values, dtype=float)

        with PdfPages(combined_pdf) as pdf:
            for region_name in names:
                ds_region = get_region_ds(ds, PREDICTION_REGION_BOXES[region_name])
                region_lon = np.asarray(ds_region["lon_hres"].values, dtype=float)
                region_lat = np.asarray(ds_region["lat_hres"].values, dtype=float)

                msl_fields = [
                    _msl_hpa(ds_region, "x_interp"),
                    _msl_hpa(ds_region, "y"),
                    _msl_hpa(ds_region, "y_pred"),
                ]
                wind_fields = [
                    _wind10m(ds_region, "x_interp"),
                    _wind10m(ds_region, "y"),
                    _wind10m(ds_region, "y_pred"),
                ]
                msl_levels = _levels(msl_fields)
                wind_levels = _levels(wind_fields)

                fig, axs = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
                titles = ["x_interp", "y", "y_pred"]
                top_mappables = []
                bottom_mappables = []
                for col, title in enumerate(titles):
                    top_mappables.append(
                        _draw_panel(
                            axs[0, col],
                            region_lon,
                            region_lat,
                            msl_fields[col],
                            levels=msl_levels,
                            title=f"{title} - msl",
                        )
                    )
                    bottom_mappables.append(
                        _draw_panel(
                            axs[1, col],
                            region_lon,
                            region_lat,
                            wind_fields[col],
                            levels=wind_levels,
                            title=f"{title} - wind10m",
                        )
                    )
                for ax in axs[:, 0]:
                    ax.set_ylabel("Latitude (°)", fontsize=10)
                for ax in axs[1, :]:
                    ax.set_xlabel("Longitude (°)", fontsize=10)

                cbar_top = fig.colorbar(top_mappables[0], ax=axs[0, :], orientation="horizontal", fraction=0.04, pad=0.07, aspect=45)
                cbar_top.set_label("MSLP (hPa)", fontsize=10)
                cbar_bottom = fig.colorbar(bottom_mappables[0], ax=axs[1, :], orientation="horizontal", fraction=0.04, pad=0.07, aspect=45)
                cbar_bottom.set_label("Wind10m (m/s)", fontsize=10)

                fig.suptitle(sample_meta_title(ds_region, region_name, sample_index) + " | contour_tc", fontsize=13, y=0.99)
                fig.tight_layout(rect=[0, 0.03, 1, 0.96])
                pdf.savefig(fig)
                region_pdf = out_root / f"{region_name}.pdf"
                fig.savefig(region_pdf, dpi=RENDER_DPI)
                generated.append(str(region_pdf))
                if also_png:
                    region_png = out_root / f"{region_name}.png"
                    fig.savefig(region_png, dpi=RENDER_DPI)
                    generated.append(str(region_png))
                plt.close(fig)

    manifest_path = _write_manifest(
        out_root=out_root,
        predictions_path=predictions_path,
        region_names=names,
        sample_index=sample_index,
        ensemble_member_index=ensemble_member_index,
        generated=generated,
    )
    generated.append(str(manifest_path))
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TC contour suites from predictions_*.nc files.")
    parser.add_argument("--predictions-nc", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--also-png", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = render_tc_contour_suite_from_predictions_file(
        predictions_nc=args.predictions_nc,
        out_dir=args.out_dir,
        region_names=[value.strip() for value in args.regions.split(",") if value.strip()] or None,
        sample_index=args.sample_index,
        ensemble_member_index=args.ensemble_member_index,
        also_png=args.also_png,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
