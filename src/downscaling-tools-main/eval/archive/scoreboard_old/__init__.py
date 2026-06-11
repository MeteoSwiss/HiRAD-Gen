"""Scoreboard scoring library — pure metric computation (JSON → scores).

Public API re-exports from submodules.
"""

from __future__ import annotations

from eval._backends.scoreboard.canonical_data import load_canonical_analysis
from eval._backends.scoreboard.row_matching import RowClassification, classify_row, find_model_row
from eval._backends.scoreboard.spectra import (
    load_spectra_metrics,
    relative_l2_weighted,
    spectra_score,
)
from eval._backends.scoreboard.surface import (
    format_surface_loss_for_scoreboard,
    load_surface_loss_metrics,
    load_x_interp_surface_metrics,
    surface_variable_nmse,
    surface_weighted_nmse,
)
from eval._backends.scoreboard.tc import (
    load_tc_extreme_scores_from_json,
    multi_depth_tc_score,
    normalize_tc_rows,
    rescale_with_eefo_floor,
)
from eval._backends.scoreboard.types import (
    SpectraMetrics,
    SurfaceMetrics,
    TCScoreResult,
)

__all__ = [
    "RowClassification",
    "SpectraMetrics",
    "SurfaceMetrics",
    "TCScoreResult",
    "classify_row",
    "find_model_row",
    "format_surface_loss_for_scoreboard",
    "load_canonical_analysis",
    "load_spectra_metrics",
    "load_surface_loss_metrics",
    "load_tc_extreme_scores_from_json",
    "load_x_interp_surface_metrics",
    "multi_depth_tc_score",
    "normalize_tc_rows",
    "relative_l2_weighted",
    "rescale_with_eefo_floor",
    "spectra_score",
    "surface_variable_nmse",
    "surface_weighted_nmse",
]
