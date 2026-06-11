"""Surface loss scoring — weighted nMSE computation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._surface_compute import SURFACE_NORMALIZATION_SCHEME
from eval._backends.scoreboard._utils import finite_float as _finite_float, load_json as _load_json
SURFACE_VAR_LABELS = {
    "10u": "10u",
    "10v": "10v",
    "2d": "2d",
    "2t": "2t",
    "msl": "MSLP",
    "skt": "SKT",
    "sp": "SP",
    "tcw": "TCW",
}



def _surface_truth_std_map_from_variables(variables: dict[str, Any]) -> dict[str, float]:
    truth_std_by_variable: dict[str, float] = {}
    for variable, entry in variables.items():
        if not isinstance(entry, dict):
            continue
        truth_std = _finite_float(entry.get("truth_std"))
        if truth_std is None or truth_std <= 0.0:
            continue
        truth_std_by_variable[variable] = truth_std
    return truth_std_by_variable


def surface_weighted_nmse(
    variables: dict[str, Any],
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> tuple[float | None, dict[str, float]]:
    per_variable_nmse: dict[str, float] = {}
    total = 0.0
    any_value = False
    for variable, entry in variables.items():
        if not isinstance(entry, dict):
            continue
        normalized_weight = _finite_float(entry.get("normalized_weight"))
        if normalized_weight is None:
            continue
        mean_nmse = _finite_float(entry.get("mean_nmse"))
        if mean_nmse is None:
            mean_mse = _finite_float(entry.get("mean_mse"))
            truth_std = None
            if truth_std_by_variable is not None:
                truth_std = _finite_float(truth_std_by_variable.get(variable))
            if mean_mse is not None and truth_std is not None and truth_std > 0.0:
                mean_nmse = mean_mse / (truth_std * truth_std)
        if mean_nmse is None or not math.isfinite(mean_nmse):
            continue
        per_variable_nmse[variable] = float(mean_nmse)
        total += float(mean_nmse) * normalized_weight
        any_value = True
    return (float(total) if any_value else None), per_variable_nmse


def load_x_interp_surface_metrics(
    predictions_dir: Path,
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> dict[str, Any]:
    from eval._backends.scoreboard._surface_compute import process_predictions_dir

    metrics = process_predictions_dir(
        predictions_dir,
        prediction_var="x_interp",
        truth_var="y",
    )
    variables = metrics.get("variables", {})
    if not isinstance(variables, dict):
        variables = {}
    local_truth_std = _surface_truth_std_map_from_variables(variables)
    weighted_nmse = _finite_float(metrics.get("weighted_surface_nmse"))
    if weighted_nmse is None and truth_std_by_variable is not None:
        weighted_nmse, per_variable_nmse = surface_weighted_nmse(
            variables,
            truth_std_by_variable=truth_std_by_variable,
        )
        for variable, value in per_variable_nmse.items():
            entry = variables.get(variable)
            if isinstance(entry, dict) and "mean_nmse" not in entry:
                entry["mean_nmse"] = value
    return {
        "weighted_mse": _finite_float(metrics.get("weighted_surface_mse")),
        "weighted_nmse": weighted_nmse,
        "normalization_scheme": str(metrics.get("normalization_scheme", SURFACE_NORMALIZATION_SCHEME)),
        "truth_std_by_variable": local_truth_std,
        "variables": variables,
        "source_path": str(predictions_dir),
    }


def load_surface_loss_metrics(
    surface_json_path: Path,
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> dict[str, Any]:
    data = _load_json(surface_json_path)
    weighted_mse = _finite_float(data.get("weighted_surface_mse"))
    variables = data.get("variables")
    variable_map = variables if isinstance(variables, dict) else {}
    local_truth_std = _surface_truth_std_map_from_variables(variable_map)
    if truth_std_by_variable is None and local_truth_std:
        truth_std_by_variable = local_truth_std
    weighted_nmse = _finite_float(data.get("weighted_surface_nmse"))
    per_variable_nmse: dict[str, float] = {}
    if weighted_nmse is None and variable_map:
        weighted_nmse, per_variable_nmse = surface_weighted_nmse(
            variable_map,
            truth_std_by_variable=truth_std_by_variable,
        )
        for variable, value in per_variable_nmse.items():
            entry = variable_map.get(variable)
            if isinstance(entry, dict) and "mean_nmse" not in entry:
                entry["mean_nmse"] = value
    normalization_scheme = str(data.get("normalization_scheme", "")).strip()
    if not normalization_scheme and weighted_nmse is not None:
        normalization_scheme = SURFACE_NORMALIZATION_SCHEME
    top_contributors: list[dict[str, Any]] = []
    if variable_map:
        contributions: list[dict[str, Any]] = []
        use_normalized_components = weighted_nmse is not None
        for variable, entry in variable_map.items():
            if not isinstance(entry, dict):
                continue
            normalized_weight = _finite_float(entry.get("normalized_weight"))
            if normalized_weight is None:
                continue
            weighted_component: float | None = None
            if use_normalized_components:
                mean_nmse = _finite_float(entry.get("mean_nmse"))
                if mean_nmse is None:
                    mean_nmse = per_variable_nmse.get(variable)
                if mean_nmse is not None and math.isfinite(mean_nmse):
                    weighted_component = mean_nmse * normalized_weight
            if weighted_component is None:
                mean_mse = _finite_float(entry.get("mean_mse"))
                if mean_mse is None:
                    continue
                weighted_component = mean_mse * normalized_weight
            contributions.append({
                "variable": variable,
                "label": SURFACE_VAR_LABELS.get(variable, variable),
                "weighted_component": weighted_component,
            })
        contributions.sort(key=lambda item: item["weighted_component"], reverse=True)
        if weighted_mse is None:
            weighted_mse = sum(item["weighted_component"] for item in contributions)
        total_source = weighted_nmse if use_normalized_components else weighted_mse
        total = total_source if total_source is not None and total_source > 0.0 else None
        for item in contributions[:2]:
            contributor = dict(item)
            share = contributor["weighted_component"] / total if total is not None else None
            if share is not None and math.isfinite(share):
                contributor["share"] = share
            top_contributors.append(contributor)
    return {
        "weighted_mse": weighted_mse,
        "weighted_nmse": weighted_nmse,
        "normalization_scheme": normalization_scheme,
        "truth_std_by_variable": truth_std_by_variable or {},
        "variables": variable_map,
        "top_contributors": top_contributors,
        "source_path": str(surface_json_path),
    }


def format_surface_loss_for_scoreboard(surface_metrics: dict[str, Any]) -> str:
    weighted_nmse = _finite_float(
        surface_metrics.get("weighted_nmse", surface_metrics.get("weighted_surface_nmse"))
    )
    if weighted_nmse is not None:
        return f"{weighted_nmse:.4f}"
    weighted_mse = _finite_float(
        surface_metrics.get("weighted_mse", surface_metrics.get("surface_weighted_mse"))
    )
    if weighted_mse is None:
        return "na"
    return f"{weighted_mse:.6e}"


def surface_variable_nmse(surface_metrics: dict[str, Any], variable: str) -> float | None:
    """Extract per-variable nMSE from surface metrics, with truth-std fallback."""
    variables = surface_metrics.get("variables")
    if not isinstance(variables, dict):
        return None
    entry = variables.get(variable)
    if not isinstance(entry, dict):
        return None
    mean_nmse = _finite_float(entry.get("mean_nmse"))
    if mean_nmse is not None:
        return mean_nmse
    mean_mse = _finite_float(entry.get("mean_mse"))
    if mean_mse is None:
        return None
    truth_std = _finite_float(entry.get("truth_std"))
    if truth_std is None:
        truth_std_map = surface_metrics.get("truth_std_by_variable")
        if isinstance(truth_std_map, dict):
            truth_std = _finite_float(truth_std_map.get(variable))
    if truth_std is None or truth_std <= 0.0:
        return None
    return mean_mse / (truth_std * truth_std)


def load_surface_weighted_mse(surface_json_path: Path) -> float | None:
    return _finite_float(load_surface_loss_metrics(surface_json_path).get("weighted_mse"))
