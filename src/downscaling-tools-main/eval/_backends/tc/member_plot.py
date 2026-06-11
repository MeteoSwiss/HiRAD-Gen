"""Member spatial maps — per-member 2x3 grid pages (MSLP/Wind x Input/Prediction/Target)."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs
from matplotlib.backends.backend_pdf import PdfPages

from .data_types import BoundingBox
from .events import TCEvent
from .experiment_config import TCExperimentConfig
from .plot_config import TCPlotConfig

LOG = logging.getLogger(__name__)


def _select_projection(bbox: BoundingBox):
    """Pick LambertConformal for non-dateline-crossing bboxes, PlateCarree otherwise."""
    if bbox.crosses_dateline:
        return crs.PlateCarree()
    central_lon = (bbox.west + bbox.east) / 2.0
    central_lat = (bbox.south + bbox.north) / 2.0
    return crs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )


def _source_labels(exp_config: TCExperimentConfig | None) -> tuple[str, str]:
    """Derive descriptive labels for input and target columns."""
    if exp_config is None:
        return "Input", "Target"

    def _fmt(expid: str) -> str:
        return expid.rsplit("_", 1)[0].replace("_", " ")

    refs = exp_config.reference_expids
    if len(refs) >= 2:
        # Convention: first reference is the "target" (higher res), second is "input" (lower res)
        input_src = _fmt(refs[-1])
        target_src = _fmt(refs[0])
    elif len(refs) == 1:
        input_src = _fmt(refs[0])
        if exp_config.analysis_expid:
            analysis_res = exp_config.analysis_expid.split("_")[1]
            target_src = f"IEKM {analysis_res}"
        else:
            target_src = "Target"
    else:
        input_src = "Input"
        target_src = "Target"
    return input_src, target_src


def _plot_member_page(
    fields: dict[str, np.ndarray],
    *,
    bbox: BoundingBox,
    plot_config: TCPlotConfig,
    exp_config: TCExperimentConfig | None,
    member_idx: int,
    member_label: int,
    step_hours: int,
    date_str: str,
    display_label: str,
    event_name: str,
) -> plt.Figure:
    """Create a 2x3 grid for one member: rows=[MSLP, Wind], cols=[Input, Prediction, Target]."""
    import cartopy.feature as cfeature
    import cmcrameri.cm as cmc
    from matplotlib.gridspec import GridSpec

    proj = _select_projection(bbox)
    input_src, target_src = _source_labels(exp_config)

    bbox_aspect = abs(bbox.east - bbox.west) / (bbox.north - bbox.south)
    fig_height = 13.0 if bbox_aspect <= 1.0 else 13.0 / bbox_aspect

    fig = plt.figure(figsize=(18, fig_height))
    gs = GridSpec(
        2, 3,
        hspace=0.35, wspace=0.12,
        left=0.08, right=0.97, bottom=0.08, top=0.91,
    )

    map_axes = np.empty((2, 3), dtype=object)
    for row_i in range(2):
        for col_i in range(3):
            map_axes[row_i, col_i] = fig.add_subplot(gs[row_i, col_i], projection=proj)

    lat = fields["lat_axis"]
    lon = fields["lon_axis"]

    col_defs = [
        ("x_interp", f"{input_src} (input)"),
        ("y_pred", "Prediction (downscaled input)"),
        ("y", f"{target_src} (target)"),
    ]

    row_defs = [
        ("msl", "MSLP", cmc.vik, "hPa"),
        ("wind", "10m Wind Speed", cmc.batlow, "m/s"),
    ]

    # Shared color limits
    msl_keys = [k for k in ["x_interp_msl", "y_pred_msl", "y_msl"] if k in fields]
    wind_keys = [k for k in ["x_interp_wind", "y_pred_wind", "y_wind"] if k in fields]
    if plot_config.member_map_msl_range is not None:
        msl_vmin, msl_vmax = plot_config.member_map_msl_range
    else:
        msl_vmin = float(np.nanmin([np.nanmin(fields[k][member_idx]) for k in msl_keys]))
        msl_vmax = float(np.nanmax([np.nanmax(fields[k][member_idx]) for k in msl_keys]))
    if plot_config.member_map_wind_range is not None:
        wind_vmin, wind_vmax = plot_config.member_map_wind_range
    else:
        wind_vmin = 0.0
        wind_vmax = float(np.nanmax([np.nanmax(fields[k][member_idx]) for k in wind_keys]))
    var_vlims = {"msl": (msl_vmin, msl_vmax), "wind": (wind_vmin, wind_vmax)}

    row_images = {}

    for row_i, (var_suffix, row_label, cmap, unit) in enumerate(row_defs):
        vmin, vmax = var_vlims[var_suffix]
        contour_levels = np.linspace(vmin, vmax, 12)

        for col_i, (src_prefix, col_title) in enumerate(col_defs):
            ax = map_axes[row_i, col_i]
            field_key = f"{src_prefix}_{var_suffix}"

            if field_key not in fields:
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="gray",
                )
                if row_i == 0:
                    ax.set_title(col_title, fontsize=13)
                ax.coastlines(linewidth=0.5)
                continue

            arr = fields[field_key][member_idx]

            im = ax.pcolormesh(
                lon, lat, arr,
                transform=crs.PlateCarree(),
                vmin=vmin, vmax=vmax,
                shading="nearest",
                cmap=cmap,
            )
            try:
                ax.contour(
                    lon, lat, arr,
                    transform=crs.PlateCarree(),
                    levels=contour_levels,
                    colors="black",
                    linewidths=0.4,
                    alpha=0.6,
                )
            except Exception:
                pass

            if row_i not in row_images:
                row_images[row_i] = im

            ax.coastlines(linewidth=0.6)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
            except Exception:
                pass

            ax.set_extent(
                [bbox.west, bbox.east, bbox.south, bbox.north],
                crs=crs.PlateCarree(),
            )

            gl = ax.gridlines(
                draw_labels=True, dms=False,
                x_inline=False, y_inline=False,
                linewidth=0.3, alpha=0.5,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"rotation": 0, "fontsize": 11, "va": "top"}
            gl.ylabel_style = {"fontsize": 11}
            gl.xpadding = 8
            if col_i > 0:
                gl.left_labels = False

            if row_i == 0:
                ax.set_title(col_title, fontsize=14)
            if col_i == 0:
                ax.text(
                    -0.14, 0.5, row_label,
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=14, rotation=90,
                )

    # Colorbars
    fig.canvas.draw()
    for row_i, (var_suffix, row_label, cmap, unit) in enumerate(row_defs):
        if row_i not in row_images:
            continue
        pos0 = map_axes[row_i, 0].get_position()
        pos2 = map_axes[row_i, 2].get_position()
        row_width = pos2.x1 - pos0.x0
        cbar_w = row_width * 0.6
        cbar_x = pos0.x0 + row_width * 0.2
        cbar_y = pos0.y0 - 0.05
        cax = fig.add_axes([cbar_x, cbar_y, cbar_w, 0.015])
        cbar = fig.colorbar(row_images[row_i], cax=cax, orientation="horizontal")
        cbar.set_label(unit, fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"TC {event_name.capitalize()} | {display_label} | {date_str} +{step_hours}h | member {member_label}",
        fontsize=16, y=0.96,
    )
    return fig
