"""Region plot evaluator — plotting-specific config (colormaps, variable rendering).

Keeps rendering details that don't belong in lane YAML config.
"""
from __future__ import annotations

DEFAULT_WEATHER_STATES = [
    "10u", "10v", "2t", "msl", "tp", "z_500", "u_850", "v_850", "t_850",
]

DEFAULT_MODEL_VARIABLES = [
    "x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0",
]

DEFAULT_COLORMAPS = {
    "default": "viridis",
    "residual": "bwr",
}
