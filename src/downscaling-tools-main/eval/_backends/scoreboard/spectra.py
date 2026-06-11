"""Spectra scoring — relative L2 distance computation and metric loading."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._utils import finite_float as _finite_float, load_json as _load_json
import numpy as np

SPECTRA_FIELDS = ("10u", "10v", "2t", "msl", "t_850", "z_500")
RAW_FIELD_DIRS = {
    "10u": "10u_sfc",
    "10v": "10v_sfc",
    "2t": "2t_sfc",
    "msl": "msl_sfc",
    "t_850": "t_850",
    "z_500": "z_500",
}
SPECTRA_FIELD_DIR_ALIASES = {
    "msl": ("sp_sfc",),
}
SPECTRA_SUMMARY_ALIASES = {
    "msl": ("sp",),
}
AMP_FILE_RE = re.compile(r"^ampl_(\d{8})_(\d+)_([a-z0-9]+)_([a-z0-9]+)_([^_]+)_n(\d+)\.npy$")
SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE = 100.0


def spectra_score(relative_l2: float) -> float:
    """Convert a relative L2 distance to a 0–1 score (higher = better)."""
    return max(0.0, 1.0 - relative_l2)



def empty_spectra_metrics(source_path: str = "na") -> dict[str, Any]:
    result: dict[str, Any] = {field: None for field in SPECTRA_FIELDS}
    result.update({
        "mean": None,
        "source_path": source_path,
        "coverage": "missing",
        "n_curves": None,
        "count_label": "pairs",
        "counts": {},
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


def _default_wavenumbers(length: int) -> np.ndarray:
    return np.arange(1, length + 1, dtype=np.float64)


def finite_positive_mask(arr: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> np.ndarray:
    wvn = _default_wavenumbers(arr.shape[0]) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    if wvn.shape != arr.shape:
        raise ValueError(f"wavenumber length mismatch: curve={arr.shape[0]} wavenumbers={wvn.shape[0]}")
    threshold = SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE
    max_wvn = float(wvn.max()) if wvn.size > 0 else 0.0
    if max_wvn <= threshold:
        threshold = max_wvn / 3.0
    return np.isfinite(arr) & (arr > 0.0) & np.isfinite(wvn) & (wvn > threshold)


def relative_l2(pred: np.ndarray, truth: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> float:
    keep = finite_positive_mask(pred, wavenumbers=wavenumbers) & finite_positive_mask(truth, wavenumbers=wavenumbers)
    if not np.any(keep):
        return float("nan")
    return float(np.linalg.norm((pred - truth)[keep]) / max(np.linalg.norm(truth[keep]), 1e-12))


def relative_l2_weighted(pred: np.ndarray, truth: np.ndarray, *, wavenumbers: np.ndarray | None = None) -> float:
    """Fine-scale-weighted relative L2: wavenumber/max_wavenumber weighting."""
    wvn = _default_wavenumbers(pred.shape[0]) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    keep = finite_positive_mask(pred, wavenumbers=wvn) & finite_positive_mask(truth, wavenumbers=wvn)
    if not np.any(keep):
        return float("nan")
    wn = wvn[keep]
    weights = wn / wn.max()
    diff = (pred - truth)[keep] * weights
    ref = truth[keep] * weights
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))


# --- Spectra file/directory helpers ---

def spectra_field_root(root: Path, field_dir: str) -> Path:
    direct = root / field_dir
    if direct.exists():
        return direct
    nested = root / "spectra" / field_dir
    if nested.exists():
        return nested
    return direct


def spectra_field_dir_candidates(field: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for field_dir in (RAW_FIELD_DIRS[field], *SPECTRA_FIELD_DIR_ALIASES.get(field, ())):
        if field_dir in seen:
            continue
        seen.add(field_dir)
        ordered.append(field_dir)
    return tuple(ordered)


def spectra_summary_keys(field: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for key in (field, *SPECTRA_SUMMARY_ALIASES.get(field, ())):
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def _resolve_spectra_field_root(root: Path, field: str) -> tuple[Path, str]:
    for field_dir in spectra_field_dir_candidates(field):
        candidate = spectra_field_root(root, field_dir)
        if candidate.exists():
            return candidate, field_dir
    fallback_dir = spectra_field_dir_candidates(field)[0]
    return spectra_field_root(root, fallback_dir), fallback_dir


def _extract_spectra_summary_entry(source: dict[str, Any], field: str) -> tuple[dict[str, Any], str | None]:
    for key in spectra_summary_keys(field):
        entry = source.get(key)
        if isinstance(entry, dict):
            return entry, key
    return {}, None


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _load_spectra_metadata(spectra_dir: Path) -> dict[str, Any]:
    staging_path = spectra_dir / "staging_summary.json"
    if staging_path.exists():
        return _load_json(staging_path)
    summary_path = spectra_dir / "spectra_summary.json"
    if summary_path.exists():
        data = _load_json(summary_path)
        if any(key in data for key in ("dates", "steps_hours", "ensemble_members", "template_root")):
            return data
    return {}



def _infer_spectra_token(spectra_dir: Path) -> str:
    seen: set[str] = set()
    for field in SPECTRA_FIELDS:
        for field_dir in spectra_field_dir_candidates(field):
            if field_dir in seen:
                continue
            seen.add(field_dir)
            for ampl_path in spectra_field_root(spectra_dir, field_dir).glob("ampl_*.npy"):
                match = AMP_FILE_RE.match(ampl_path.name)
                if match:
                    return match.group(5)
    return "1"


def _has_raw_spectra_arrays(spectra_dir: Path) -> bool:
    seen: set[str] = set()
    for field in SPECTRA_FIELDS:
        for field_dir in spectra_field_dir_candidates(field):
            if field_dir in seen:
                continue
            seen.add(field_dir)
            if any(spectra_field_root(spectra_dir, field_dir).glob("ampl_*.npy")):
                return True
    return False


def _load_curve_wavenumbers(root: Path, *, date: int, step: int, field_dir: str, token: str, member: int) -> np.ndarray | None:
    path = root / f"wvn_{date}_{step}_{field_dir}_{token}_n{member}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=np.float64)


# --- Summary extraction ---

def _extract_spectra_metrics_from_summary(data: dict[str, Any], source_path: str) -> dict[str, Any]:
    source: dict[str, Any] | None = None
    count_label = "pairs"
    scopes = data.get("scopes")
    if isinstance(scopes, dict) and isinstance(scopes.get("residual"), dict):
        source = scopes["residual"]
        count_label = "curves"
    elif isinstance(data.get("weather_states"), dict) and data["weather_states"]:
        source = data["weather_states"]
        count_label = "curves"
    elif any(isinstance(data.get(field), dict) for field in SPECTRA_FIELDS):
        source = data
    if source is None:
        return empty_spectra_metrics(source_path)

    scores: dict[str, float | None] = {}
    values: list[float] = []
    curve_counts: list[int] = []
    counts: dict[str, int] = {}
    missing_reference_pairs: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        entry, _ = _extract_spectra_summary_entry(source, field)
        if not entry:
            scores[field] = None
            continue
        value = _finite_float(entry.get("relative_l2_mean_curve"))
        if value is not None:
            scores[field] = value
            values.append(value)
        else:
            scores[field] = None
        count = entry.get("n_curves")
        if count is None:
            count = entry.get("n_pairs")
        if isinstance(count, int):
            curve_counts.append(int(count))
            counts[field] = int(count)
        missing = entry.get("missing_reference_pairs")
        if isinstance(missing, int):
            missing_reference_pairs[field] = int(missing)

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": source_path,
        "coverage": f"{curve_counts[0]} {count_label}" if curve_counts and len(set(curve_counts)) == 1 else f"mixed {count_label}",
        "n_curves": curve_counts[0] if curve_counts and len(set(curve_counts)) == 1 else None,
        "count_label": count_label,
        "counts": counts,
        "missing_reference_pairs": missing_reference_pairs,
        "score_wavenumber_min_exclusive": _finite_float(data.get("score_wavenumber_min_exclusive"))
        or SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


# --- Raw spectra loading ---

def _load_raw_spectra_metrics(spectra_dir: Path, reference_root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    dates = _int_list(metadata.get("dates"))
    steps = _int_list(metadata.get("steps_hours", metadata.get("steps")))
    members = _int_list(metadata.get("ensemble_members", metadata.get("members")))
    if not dates or not steps or not members:
        return empty_spectra_metrics(str(spectra_dir))

    token = _infer_spectra_token(spectra_dir)
    ref_token = str(metadata.get("reference_token", "1")).strip() or "1"

    scores: dict[str, float | None] = {}
    values: list[float] = []
    coverage_parts: list[str] = []
    curve_count: int | None = None
    counts: dict[str, int] = {}
    missing_reference_pairs: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        exp_curves: list[np.ndarray] = []
        ref_curves: list[np.ndarray] = []
        wavenumbers: np.ndarray | None = None
        used_dates: set[int] = set()
        used_steps: set[int] = set()
        used_members: set[int] = set()
        misses = 0
        exp_dir, exp_field_dir = _resolve_spectra_field_root(spectra_dir, field)
        ref_dir, ref_field_dir = _resolve_spectra_field_root(reference_root, field)
        for date in dates:
            for step in steps:
                for member in members:
                    exp_file = exp_dir / f"ampl_{date}_{step}_{exp_field_dir}_{token}_n{member}.npy"
                    ref_file = ref_dir / f"ampl_{date}_{step}_{ref_field_dir}_{ref_token}_n{member}.npy"
                    if not exp_file.exists():
                        continue
                    if not ref_file.exists():
                        misses += 1
                        continue
                    exp_curves.append(np.load(exp_file))
                    ref_curves.append(np.load(ref_file))
                    curve_wavenumbers = _load_curve_wavenumbers(
                        exp_dir, date=date, step=step, field_dir=exp_field_dir, token=token, member=member,
                    )
                    if curve_wavenumbers is None:
                        curve_wavenumbers = _load_curve_wavenumbers(
                            ref_dir, date=date, step=step, field_dir=ref_field_dir, token=ref_token, member=member,
                        )
                    if curve_wavenumbers is not None:
                        if wavenumbers is None:
                            wavenumbers = curve_wavenumbers
                        elif curve_wavenumbers.shape != wavenumbers.shape or not np.allclose(curve_wavenumbers, wavenumbers):
                            raise ValueError(
                                f"wavenumber mismatch for field={field}: "
                                f"candidate={curve_wavenumbers.shape} reference={wavenumbers.shape} "
                                f"(root={spectra_dir}, reference_root={reference_root})"
                            )
                    used_dates.add(date)
                    used_steps.add(step)
                    used_members.add(member)

        missing_reference_pairs[field] = misses
        counts[field] = len(exp_curves)
        if not exp_curves or not ref_curves:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        exp_mean = np.nanmean(np.stack(exp_curves, axis=0), axis=0)
        ref_mean = np.nanmean(np.stack(ref_curves, axis=0), axis=0)
        if exp_mean.shape != ref_mean.shape:
            raise ValueError(
                f"spectra length mismatch for field={field}: "
                f"candidate={exp_mean.shape[0]} reference={ref_mean.shape[0]} "
                f"(root={spectra_dir}, reference_root={reference_root})"
            )
        score = relative_l2_weighted(exp_mean, ref_mean, wavenumbers=wavenumbers)
        scores[field] = score if math.isfinite(score) else None
        if scores[field] is not None:
            values.append(scores[field])
        coverage_parts.append(
            f"{field}: {len(used_dates)}d/{len(used_members)}m/{len(used_steps)}s/{len(exp_curves)}c"
        )
        if curve_count is None:
            curve_count = len(exp_curves)

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(spectra_dir),
        "coverage": "; ".join(coverage_parts) if coverage_parts else "missing",
        "n_curves": curve_count,
        "count_label": "pairs",
        "counts": counts,
        "missing_reference_pairs": missing_reference_pairs,
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


# --- Comparison summary loading ---

def _match_comparison_model(
    models: list[dict[str, Any]],
    *,
    preferred_name: str = "",
    preferred_path: Path | None = None,
) -> dict[str, Any] | None:
    target_name = preferred_name.strip()
    target_path = str(preferred_path) if preferred_path is not None else ""
    target_path_name = preferred_path.name if preferred_path is not None else ""
    for model in models:
        name = str(model.get("name", "")).strip()
        path_text = str(model.get("path", "")).strip()
        path_name = Path(path_text).name if path_text else ""
        if target_path and path_text == target_path:
            return model
        if target_name and name == target_name:
            return model
        if target_path_name and path_name == target_path_name:
            return model
    return None


def _load_step_mean_curve(root: Path, *, field_dir: str, step: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    field_root = spectra_field_root(root, field_dir)
    w_arrays: list[np.ndarray] = []
    a_arrays: list[np.ndarray] = []
    pattern = f"wvn_*_{step}_{field_dir}_*_n*.npy"
    for w_path in sorted(field_root.glob(pattern)):
        a_path = w_path.with_name(w_path.name.replace("wvn_", "ampl_", 1))
        if not a_path.exists():
            continue
        try:
            wavenumbers = np.asarray(np.load(w_path), dtype=np.float64)
            amplitudes = np.asarray(np.load(a_path), dtype=np.float64)
        except (OSError, ValueError):
            continue
        if wavenumbers.shape != amplitudes.shape or wavenumbers.size < 8:
            continue
        if w_arrays and (wavenumbers.shape != w_arrays[0].shape or amplitudes.shape != a_arrays[0].shape):
            continue
        w_arrays.append(wavenumbers)
        a_arrays.append(amplitudes)
    if not a_arrays:
        return None, None, 0
    mean_wavenumbers = np.nanmean(np.stack(w_arrays, axis=0), axis=0)
    mean_amplitudes = np.nanmean(np.stack(a_arrays, axis=0), axis=0)
    return mean_wavenumbers, mean_amplitudes, len(a_arrays)


def _load_reference_model_comparison_metrics(
    spectra_dir: Path,
    comparison_path: Path,
    *,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    data = _load_json(comparison_path)
    models = data.get("models")
    per_param = data.get("per_param")
    if not isinstance(models, list) or not isinstance(per_param, dict):
        return empty_spectra_metrics(str(comparison_path))

    model_entries = [entry for entry in models if isinstance(entry, dict)]
    candidate_model = _match_comparison_model(
        model_entries, preferred_name=spectra_dir.name, preferred_path=spectra_dir,
    )
    if candidate_model is None:
        return empty_spectra_metrics(str(comparison_path))

    reference_model = None
    if reference_root is not None:
        reference_model = _match_comparison_model(
            model_entries, preferred_name=reference_root.name, preferred_path=reference_root,
        )
    if reference_model is None:
        reference_model = _match_comparison_model(model_entries, preferred_name="enfo_o320")
    if reference_model is None:
        return empty_spectra_metrics(str(comparison_path))

    candidate_name = str(candidate_model.get("name", "")).strip()
    reference_name = str(reference_model.get("name", "")).strip()
    candidate_root = Path(str(candidate_model.get("path", "")).strip())
    reference_model_root = Path(str(reference_model.get("path", "")).strip())
    if not candidate_name or not reference_name or not candidate_root.exists() or not reference_model_root.exists():
        return empty_spectra_metrics(str(comparison_path))

    scores: dict[str, float | None] = {}
    values: list[float] = []
    counts: dict[str, int] = {}
    coverage_parts: list[str] = []
    curve_count: int | None = None
    for field in SPECTRA_FIELDS:
        field_dir = None
        candidate_entry: dict[str, Any] | None = None
        reference_entry: dict[str, Any] | None = None
        for candidate_field_dir in spectra_field_dir_candidates(field):
            param_entry = per_param.get(candidate_field_dir)
            if not isinstance(param_entry, dict):
                continue
            trial_candidate = param_entry.get(candidate_name)
            trial_reference = param_entry.get(reference_name)
            if isinstance(trial_candidate, dict) and isinstance(trial_reference, dict):
                field_dir = candidate_field_dir
                candidate_entry = trial_candidate
                reference_entry = trial_reference
                break
        if field_dir is None or candidate_entry is None or reference_entry is None:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        candidate_step = candidate_entry.get("step")
        reference_step = reference_entry.get("step")
        candidate_status = str(candidate_entry.get("status", "")).strip().lower()
        reference_status = str(reference_entry.get("status", "")).strip().lower()
        if (
            not isinstance(candidate_step, int)
            or not isinstance(reference_step, int)
            or candidate_status != "plotted"
            or reference_status != "plotted"
        ):
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue

        candidate_wvn, candidate_mean, candidate_count = _load_step_mean_curve(
            candidate_root, field_dir=field_dir, step=int(candidate_step),
        )
        _, reference_mean, reference_count = _load_step_mean_curve(
            reference_model_root, field_dir=field_dir, step=int(reference_step),
        )
        counts[field] = min(candidate_count, reference_count)
        if candidate_mean is None or reference_mean is None:
            scores[field] = None
            coverage_parts.append(f"{field}: missing")
            continue
        if candidate_mean.shape != reference_mean.shape:
            raise ValueError(
                f"spectra length mismatch for field={field}: "
                f"candidate={candidate_mean.shape[0]} reference={reference_mean.shape[0]} "
                f"(root={candidate_root}, reference_root={reference_model_root})"
            )
        score = relative_l2_weighted(candidate_mean, reference_mean, wavenumbers=candidate_wvn)
        scores[field] = score if math.isfinite(score) else None
        if scores[field] is not None:
            values.append(scores[field])
        coverage_parts.append(
            f"{field}: step{candidate_step}/{reference_step} {candidate_count}c/{reference_count}r"
        )
        if curve_count is None and candidate_count == reference_count:
            curve_count = candidate_count

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(comparison_path),
        "coverage": "; ".join(coverage_parts) if coverage_parts else "missing",
        "n_curves": curve_count,
        "count_label": "curves",
        "counts": counts,
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


# --- Curve summary rescoring ---

def _rescore_from_curve_summary(spectra_dir: Path) -> dict[str, Any]:
    """Rescore from spectra_curve_summary.json using adaptive threshold."""
    curve_path = spectra_dir / "spectra_curve_summary.json"
    if not curve_path.exists():
        return empty_spectra_metrics(str(spectra_dir))
    data = _load_json(curve_path)
    ws = data.get("weather_states", {})
    if not ws:
        return empty_spectra_metrics(str(curve_path))

    scores: dict[str, float | None] = {}
    values: list[float] = []
    counts: dict[str, int] = {}
    for field in SPECTRA_FIELDS:
        entry, _ = _extract_spectra_summary_entry(ws, field)
        if not entry:
            scores[field] = None
            continue
        scopes = entry.get("scopes", {})
        scope = scopes.get("residual") or scopes.get("full_field") or entry
        wvn_raw = scope.get("wavenumbers")
        pred_raw = scope.get("prediction_mean")
        truth_raw = scope.get("truth_mean")
        if not wvn_raw or not pred_raw or not truth_raw:
            scores[field] = None
            continue
        wvn = np.asarray(wvn_raw, dtype=np.float64)
        pred = np.asarray(pred_raw, dtype=np.float64)
        truth = np.asarray(truth_raw, dtype=np.float64)
        if wvn.shape != pred.shape or wvn.shape != truth.shape:
            scores[field] = None
            continue
        score = relative_l2_weighted(pred, truth, wavenumbers=wvn)
        score_val = _finite_float(score)
        scores[field] = score_val
        if score_val is not None:
            values.append(score_val)
        n = scope.get("n_curves")
        if isinstance(n, int):
            counts[field] = n

    available_count = len(values)
    mean = sum(values) / available_count if available_count >= 3 else None
    n_curves_list = list(counts.values())
    result: dict[str, Any] = {field: scores.get(field) for field in SPECTRA_FIELDS}
    result.update({
        "mean": mean,
        "source_path": str(curve_path),
        "coverage": f"{n_curves_list[0]} curves" if n_curves_list and len(set(n_curves_list)) == 1 else "mixed curves",
        "n_curves": n_curves_list[0] if n_curves_list and len(set(n_curves_list)) == 1 else None,
        "count_label": "curves",
        "counts": counts,
        "missing_reference_pairs": {},
        "score_wavenumber_min_exclusive": _finite_float(data.get("score_wavenumber_min_exclusive"))
        or SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    })
    return result


# --- Main entry point ---

def load_spectra_metrics(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    summary_path = spectra_dir / "spectra_summary.json"
    comparison_path = spectra_dir / "comparison_summary.json"
    summary_data: dict[str, Any] = {}
    if summary_path.exists():
        summary_data = _load_json(summary_path)
    metadata = _load_spectra_metadata(spectra_dir)
    ref_path: Path | None = reference_root
    if ref_path is None:
        relative_to = str(summary_data.get("relative_to", "")).strip()
        if relative_to:
            ref_path = Path(relative_to)
    if ref_path is None:
        template_root = str(metadata.get("template_root", "")).strip()
        if template_root:
            ref_path = Path(template_root)
    if ref_path is not None and ref_path.exists() and _has_raw_spectra_arrays(spectra_dir):
        raw_metrics = _load_raw_spectra_metrics(spectra_dir, ref_path, metadata)
        if raw_metrics["mean"] is not None or any(raw_metrics[field] is not None for field in SPECTRA_FIELDS):
            return raw_metrics
    summary_metrics = empty_spectra_metrics(str(summary_path if summary_path.exists() else spectra_dir))
    if summary_path.exists():
        summary_metrics = _extract_spectra_metrics_from_summary(summary_data, str(summary_path))
        if summary_metrics["mean"] is not None or any(summary_metrics[field] is not None for field in SPECTRA_FIELDS):
            return summary_metrics
    if comparison_path.exists():
        comparison_metrics = _load_reference_model_comparison_metrics(
            spectra_dir, comparison_path, reference_root=ref_path,
        )
        if comparison_metrics["mean"] is not None or any(comparison_metrics[field] is not None for field in SPECTRA_FIELDS):
            return comparison_metrics
    if summary_metrics["mean"] is not None or any(summary_metrics[field] is not None for field in SPECTRA_FIELDS):
        return summary_metrics
    curve_rescore = _rescore_from_curve_summary(spectra_dir)
    if curve_rescore["mean"] is not None or any(curve_rescore[field] is not None for field in SPECTRA_FIELDS):
        return curve_rescore
    return empty_spectra_metrics(str(summary_path if summary_path.exists() else spectra_dir))


def build_spectra_summary(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    metrics = load_spectra_metrics(spectra_dir, reference_root=reference_root)
    relative_to = str(reference_root or "")
    if not relative_to:
        metadata = _load_spectra_metadata(spectra_dir)
        relative_to = str(metadata.get("template_root", "")).strip()

    summary: dict[str, Any] = {
        "method": "ecmwf_mean_curve_reference_weighted_l2",
        "relative_to": relative_to,
        "spectra_output_dir": str(spectra_dir),
        "scoreboard_fields": list(SPECTRA_FIELDS),
        "score_wavenumber_min_exclusive": SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    }
    if metrics["mean"] is not None:
        summary["mean_relative_l2"] = float(metrics["mean"])
    if metrics.get("coverage"):
        summary["coverage"] = metrics["coverage"]

    counts = metrics.get("counts", {})
    missing_reference_pairs = metrics.get("missing_reference_pairs", {})
    count_label = str(metrics.get("count_label", "pairs")).strip() or "pairs"
    count_key = "n_curves" if count_label == "curves" else "n_pairs"
    for field in SPECTRA_FIELDS:
        score = metrics.get(field)
        field_summary: dict[str, Any]
        if score is None:
            field_summary = {"status": "missing"}
        else:
            field_summary = {"relative_l2_mean_curve": float(score)}
        if isinstance(counts.get(field), int):
            field_summary[count_key] = int(counts[field])
        if isinstance(missing_reference_pairs.get(field), int):
            field_summary["missing_reference_pairs"] = int(missing_reference_pairs[field])
        summary[field] = field_summary
    return summary
