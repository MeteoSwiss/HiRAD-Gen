"""Per-event visualization defaults and reference styles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCPlotConfig:
    mslp_bin_range: tuple[float, float, float] = (980, 1021, 1)
    wind_bin_range: tuple[float, float, float] = (0, 35.01, 1)
    mslp_ylim: tuple[float, float] = (0, 4)
    wind_ylim: tuple[float, float] = (0, 2)
    regrid_resolution: float = 0.25
    plot_title: str = ""
    member_map_msl_range: tuple[float, float] | None = None
    member_map_wind_range: tuple[float, float] | None = None


PLOT_CONFIGS: dict[str, TCPlotConfig] = {
    "franklin": TCPlotConfig(plot_title="Franklin normed pdfs"),
    "idalia": TCPlotConfig(plot_title="Idalia normed pdfs"),
    "franklin_idalia": TCPlotConfig(plot_title="Franklin + Idalia normed pdfs"),
    "hilary": TCPlotConfig(
        mslp_bin_range=(960, 1021, 1),
        wind_bin_range=(0, 30.01, 1),
        mslp_ylim=(0, 2),
        plot_title="Hilary normed pdfs",
    ),
    "dora": TCPlotConfig(
        mslp_bin_range=(970, 1021, 1),
        plot_title="Dora normed pdfs",
    ),
    "fernanda": TCPlotConfig(
        wind_bin_range=(0, 30.01, 1),
        mslp_ylim=(0, 10),
        wind_ylim=(0, 5),
        plot_title="Fernanda normed pdfs",
    ),
    "humberto": TCPlotConfig(
        mslp_bin_range=(910, 1025, 1),
        wind_bin_range=(0, 60.01, 1),
        plot_title="TC Humberto 2025-09 | norm. PDFs (MSLP & 10m Wind)",
        member_map_msl_range=(950, 1025),
        member_map_wind_range=(0, 45),
    ),
}


REFERENCE_STYLES: dict[str, dict] = {
    "ENFO_O320_0001": {"label": "enfo_o320", "color": "black", "linestyle": "-.", "linewidth": 2},
    "ENFO_O48_0001": {"label": "enfo_o48", "color": "black", "linestyle": "-.", "linewidth": 2},
    "EEFO_O96_0001": {"label": "eefo_o96", "color": "red", "linestyle": "--", "linewidth": 2},
    "ENFO_O96_0001": {"label": "enfo_o96", "color": "red", "linestyle": "--", "linewidth": 2},
    "ENFO_O320_ip6y": {"label": "ip6y", "color": "orange", "linestyle": ":", "linewidth": 2},
    "IEKM_O96_TARGET": {"label": "iekm-o96", "color": "steelblue", "linestyle": "--", "linewidth": 2},
}
