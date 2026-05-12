"""Surface evaluator — weighted nMSE scoring.

Wraps eval.scoreboard.surface scoring functions with the standard
evaluator interface. Scoring algorithms are imported directly to
guarantee mathematical equivalence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard.surface import (
    load_surface_loss_metrics,
    load_x_interp_surface_metrics,
)
from eval._backends.scoreboard._surface_compute import process_predictions_dir

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    surface_json: str = "surface_loss.json",
    predictions_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Score surface loss from a surface JSON or by computing from predictions.

    Parameters
    ----------
    results_dir : Path to the evaluator output or predictions directory.
    lane_config : Full lane configuration dict.
    eval_config : Surface-specific config (lane_config["surface"]).
    surface_json : Name of the surface loss JSON file.
    predictions_dir : If provided, compute surface metrics directly from predictions.

    Returns
    -------
    List of {"metric": str, "value": float, "unit": str} records.
    """
    results_dir = Path(results_dir)
    weighting = eval_config.get("weighting", "truth-std")

    # Try loading from pre-computed JSON first
    json_path = results_dir / surface_json
    if json_path.exists():
        metrics = load_surface_loss_metrics(json_path)
    elif predictions_dir:
        # Compute from predictions using y_pred vs y (standard evaluation)
        raw = process_predictions_dir(Path(predictions_dir))
        metrics = {
            "weighted_mse": raw.get("weighted_surface_mse"),
            "weighted_nmse": raw.get("weighted_surface_nmse"),
            "variables": raw.get("variables", {}),
        }
    else:
        # Try the results_dir as a predictions directory
        metrics = load_x_interp_surface_metrics(results_dir)

    records: list[dict[str, Any]] = []

    weighted_nmse = metrics.get("weighted_nmse")
    if weighted_nmse is not None:
        records.append({
            "metric": "surface_weighted_nmse",
            "value": weighted_nmse,
            "unit": "nmse",
        })

    weighted_mse = metrics.get("weighted_mse")
    if weighted_mse is not None:
        records.append({
            "metric": "surface_weighted_mse",
            "value": weighted_mse,
            "unit": "mse",
        })

    # Per-variable nMSE breakdown
    variables = metrics.get("variables", {})
    if isinstance(variables, dict):
        for var_name, entry in variables.items():
            if not isinstance(entry, dict):
                continue
            nmse = entry.get("mean_nmse")
            if nmse is not None:
                records.append({
                    "metric": f"surface_{var_name}_nmse",
                    "value": nmse,
                    "unit": "nmse",
                })

    return records
