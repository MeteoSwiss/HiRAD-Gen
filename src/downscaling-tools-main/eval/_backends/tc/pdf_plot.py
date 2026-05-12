"""PDF ratio visualization — matplotlib rendering only."""
from __future__ import annotations

import logging
import warnings

import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .plot_config import REFERENCE_STYLES, TCPlotConfig
from .stats import safe_ratio

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)

sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
np.seterr(divide="ignore", invalid="ignore")


def curve_label(curve_key: str, exp_labels: dict[str, str], *, oper_key: str) -> str:
    if curve_key in REFERENCE_STYLES:
        return REFERENCE_STYLES[curve_key]["label"]
    if curve_key == oper_key:
        return "OPER AN"
    if curve_key in exp_labels:
        return exp_labels[curve_key]
    return curve_key.replace("ENFO_O320_", "")


def curve_style(
    curve_key: str,
    *,
    ml_palette: np.ndarray,
    ml_index: int,
) -> dict[str, object]:
    if curve_key in REFERENCE_STYLES:
        return dict(REFERENCE_STYLES[curve_key])
    return {
        "color": ml_palette[ml_index],
        "linestyle": "-",
        "linewidth": 3,
    }


def plot_pdf_ratios(
    plot_config: TCPlotConfig,
    *,
    event_stats: dict,
    exp_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render pre-computed event stats as a PDF ratio figure.

    Takes the output of workflows.compute_event_stats().
    """
    exp_labels = exp_labels or {}
    oper_key = event_stats["analysis_key"]
    curve_order = event_stats["curve_order"]
    var_mslp = event_stats["variables"]["mslp_hpa"]
    var_wind = event_stats["variables"]["wind10m_ms"]

    oper_hist_msl = np.asarray(var_mslp["oper_histogram"])
    oper_hist_wind = np.asarray(var_wind["oper_histogram"])
    mids_msl = np.asarray(var_mslp["bin_mids"])
    mids_wind = np.asarray(var_wind["bin_mids"])

    ml_like_keys = [k for k in curve_order if k not in REFERENCE_STYLES]
    ml_palette = cm.batlow(np.linspace(0, 1, max(1, len(ml_like_keys))))
    ml_indices = {k: idx for idx, k in enumerate(ml_like_keys)}

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for key in curve_order:
        label = curve_label(key, exp_labels, oper_key=oper_key)
        style = curve_style(key, ml_palette=ml_palette, ml_index=ml_indices.get(key, 0))

        hist_msl = np.asarray(var_mslp["curves"][key]["histogram"])
        axs[0].plot(
            mids_msl,
            safe_ratio(hist_msl, oper_hist_msl),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

        hist_wind = np.asarray(var_wind["curves"][key]["histogram"])
        axs[1].plot(
            mids_wind,
            safe_ratio(hist_wind, oper_hist_wind),
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    # Auto-crop x-axis
    xbins_msl = np.asarray(var_mslp["bin_edges"])
    xbins_wind = np.asarray(var_wind["bin_edges"])
    if "data_range_msl" in var_mslp:
        lo, hi = var_mslp["data_range_msl"]
        axs[0].set_xlim(max(xbins_msl[0], lo - 5), min(xbins_msl[-1], hi + 5))
    if "data_range_wind" in var_wind:
        lo, hi = var_wind["data_range_wind"]
        axs[1].set_xlim(0, min(xbins_wind[-1], hi + 2))

    for ax, mids, ylim, xlabel, title in [
        (axs[0], mids_msl, plot_config.mslp_ylim,
         "Mean Sea Level Pressure (hPa)", "Normalized (by analysis) Distribution MSLP"),
        (axs[1], mids_wind, plot_config.wind_ylim,
         "10m wind speed (m/s)", "Normalized (by analysis) Distribution 10m Wind Speed"),
    ]:
        ax.plot(mids, np.ones_like(mids), "--", linewidth=2, color="green", label="OPER AN")
        ydata_max = max(
            (np.nanmax(line.get_ydata()) for line in ax.get_lines() if line.get_ydata().size),
            default=0.0,
        )
        if np.isfinite(ydata_max) and ydata_max > ylim[1]:
            ax.set_yscale("symlog", linthresh=ylim[1])
            ax.set_ylim(0, None)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Normalized Probability Density", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.legend()

    fig.suptitle(plot_config.plot_title)
    fig.tight_layout()
    return fig
