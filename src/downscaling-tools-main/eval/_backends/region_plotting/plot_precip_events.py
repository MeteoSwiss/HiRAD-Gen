#!/usr/bin/env python3
"""Plot intense precipitation events from o48->o96 prediction files.

Handles two input layouts:
  --predictions-nc FILE   single predictions.nc (from-dataloader output, multiple samples)
  --predictions-dir DIR   directory of predictions_*.nc files (from-bundle output)

Outputs a PDF with one page per top event:
  row: x_interp (LR→HR) | y truth | y_pred,  zoomed ±(dlat/dlon) around event centre.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from .plotting.datetime_utils import safe_datetime_str

PRECIP_VARS = ("tp", "cp")
DEFAULT_DLAT = 18
DEFAULT_DLON = 22


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_samples(src: Path):
    """Yield (label, ds_sample) for each (sample, member=0) in src."""
    if src.is_file():
        ds = xr.open_dataset(src, engine="netcdf4")
        n = int(ds.sizes.get("sample", 1))
        for i in range(n):
            si = ds.isel(sample=i) if "sample" in ds.dims else ds
            si = si.isel(ensemble_member=0) if "ensemble_member" in si.dims else si
            date_val = si["date"].values if "date" in si else i
            yield safe_datetime_str(date_val) or str(i), si
        ds.close()
    elif src.is_dir():
        files = sorted(src.glob("predictions_*.nc"))
        if not files:
            raise FileNotFoundError(f"No predictions_*.nc in {src}")
        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            si = ds.isel(sample=0) if "sample" in ds.dims else ds
            si = si.isel(ensemble_member=0) if "ensemble_member" in si.dims else si
            date_val = si["date"].values if "date" in si else f.stem
            yield safe_datetime_str(date_val) or f.stem, si
            ds.close()
    else:
        raise FileNotFoundError(f"Not a file or directory: {src}")


def _find_precip_var(ws_values) -> str | None:
    for v in PRECIP_VARS:
        if v in ws_values:
            return v
    return None


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def _collect_events(src: Path, var: str, n_top: int) -> list[tuple[float, str, xr.Dataset, int]]:
    """Return top n_top (max_val, label, ds_sample, max_hr_idx) sorted by -max_val."""
    events: list[tuple[float, str, xr.Dataset, int]] = []
    for label, ds in _iter_samples(src):
        ws = list(ds["weather_state"].values)
        if var not in ws:
            continue
        pred = ds["y_pred"].sel(weather_state=var).values.ravel()
        max_val = float(np.nanmax(pred))
        max_idx = int(np.nanargmax(pred))
        events.append((max_val, label, ds, max_idx))
    events.sort(key=lambda e: -e[0])
    return events[:n_top]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _zoom_mask(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float,
               dlat: float, dlon: float) -> np.ndarray:
    return (
        (lat >= clat - dlat) & (lat <= clat + dlat)
        & (lon >= clon - dlon) & (lon <= clon + dlon)
    )


def _make_event_figure(
    label: str,
    ds: xr.Dataset,
    var: str,
    max_hr_idx: int,
    run_label: str,
    dlat: float,
    dlon: float,
) -> plt.Figure:
    lat_hr = ds["lat_hres"].values
    lon_hr = ds["lon_hres"].values
    clat = float(lat_hr[max_hr_idx])
    clon = float(lon_hr[max_hr_idx])

    hr_mask = _zoom_mask(lat_hr, lon_hr, clat, clon, dlat, dlon)

    y_pred_full = ds["y_pred"].sel(weather_state=var).values.ravel()
    y_pred_z = y_pred_full[hr_mask]

    has_truth = "y" in ds
    has_interp = "x_interp" in ds
    n_panels = 1 + int(has_interp) + int(has_truth)

    if has_truth:
        y_full = ds["y"].sel(weather_state=var).values.ravel()
        y_z = y_full[hr_mask]
        vmax = max(float(np.nanmax(y_pred_z)), float(np.nanmax(y_z)))
    else:
        y_z = None
        vmax = float(np.nanmax(y_pred_z))

    if has_interp:
        xi_full = ds["x_interp"].sel(weather_state=var).values.ravel()
        xi_z = xi_full[hr_mask]
        vmax = max(vmax, float(np.nanmax(xi_z)))
    else:
        xi_z = None

    vmax = max(vmax, 1e-6)
    lat_z = lat_hr[hr_mask]
    lon_z = lon_hr[hr_mask]

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panels: list[tuple[np.ndarray, str]] = []
    if xi_z is not None:
        panels.append((xi_z, f"x_interp ({var}, LR→HR)"))
    if y_z is not None:
        panels.append((y_z, f"y truth ({var})"))
    panels.append((y_pred_z, f"y_pred ({var})"))

    for ax, (data, title) in zip(axes, panels):
        sc = ax.scatter(lon_z, lat_z, c=data, s=2, cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=f"{var} (m)")
        ax.set_xlim(clon - dlon, clon + dlon)
        ax.set_ylim(clat - dlat, clat + dlat)
        ax.axhline(clat, color="r", lw=0.5, alpha=0.6)
        ax.axvline(clon, color="r", lw=0.5, alpha=0.6)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(title, fontsize=9)

    max_pred_mm = y_pred_full[max_hr_idx] * 1000
    fig.suptitle(
        f"{run_label} | {label} | event ({clat:.1f}°N, {clon:.1f}°E)"
        f" | max y_pred={max_pred_mm:.1f} mm",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _make_overview_figure(
    events: list[tuple[float, str, xr.Dataset, int]],
    var: str,
    run_label: str,
) -> plt.Figure:
    """Global scatter showing event locations coloured by max tp."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for max_val, label, ds, max_hr_idx in events:
        lat_hr = ds["lat_hres"].values
        lon_hr = ds["lon_hres"].values
        ax.plot(
            float(lon_hr[max_hr_idx]),
            float(lat_hr[max_hr_idx]),
            "o",
            markersize=8,
            label=f"{label[:10]} ({max_val * 1000:.1f} mm)",
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"{run_label} — top-{len(events)} {var} event locations (max y_pred)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot intense precipitation events.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions-nc", default="", help="Single predictions.nc file.")
    src.add_argument("--predictions-dir", default="", help="Directory of predictions_*.nc files.")
    parser.add_argument("--out", required=True, help="Output PDF path.")
    parser.add_argument("--var", default="", help="Precipitation variable (default: auto-detect tp/cp).")
    parser.add_argument("--n-top", type=int, default=6, help="Number of top events to plot.")
    parser.add_argument("--run-label", default="", help="Label shown in plot titles.")
    parser.add_argument("--dlat", type=float, default=DEFAULT_DLAT)
    parser.add_argument("--dlon", type=float, default=DEFAULT_DLON)
    args = parser.parse_args()

    src_path = Path(args.predictions_nc or args.predictions_dir)
    run_label = args.run_label or src_path.parent.name

    # Detect var from first sample if not given
    var = args.var
    if not var:
        for _, ds in _iter_samples(src_path):
            ws = list(ds["weather_state"].values)
            var = _find_precip_var(ws)
            if var:
                break
        if not var:
            raise SystemExit(f"No precipitation variable ({PRECIP_VARS}) found in predictions.")
    print(f"Precipitation variable: {var}")

    events = _collect_events(src_path, var, args.n_top)
    if not events:
        raise SystemExit(f"No samples with variable '{var}' found.")
    print(f"Top {len(events)} events by max {var} (m):")
    for mv, lbl, _, _ in events:
        print(f"  {lbl}: {mv:.6f} m = {mv * 1000:.2f} mm")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig_ov = _make_overview_figure(events, var, run_label)
        pdf.savefig(fig_ov, bbox_inches="tight")
        plt.close(fig_ov)
        for max_val, label, ds, max_hr_idx in events:
            fig = _make_event_figure(label, ds, var, max_hr_idx, run_label, args.dlat, args.dlon)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
