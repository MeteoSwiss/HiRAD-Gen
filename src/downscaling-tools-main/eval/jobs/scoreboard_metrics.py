from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._utils import finite_float, load_json
import numpy as np

# ---------------------------------------------------------------------------
# Backward-compatibility shims: scoring logic has moved to eval.scoreboard.*
# These re-exports keep existing callers working unchanged.
# ---------------------------------------------------------------------------

from eval._backends.scoreboard.canonical_data import load_canonical_analysis as _load_canonical_analysis
from eval._backends.scoreboard.row_matching import (
    classify_row as _classify_row,
    extract_checkpoint_token,
    find_model_row as _choose_tc_row,
    find_row_by_predicate as _find_row_by_predicate,
    is_analysis_row as _is_analysis_row,
    is_eefo_row as _is_eefo_row,
    is_reference_row as _is_reference_row,
    tc_candidates as _tc_candidates,
)
from eval._backends.scoreboard.spectra import (
    AMP_FILE_RE,
    RAW_FIELD_DIRS,
    SPECTRA_FIELD_DIR_ALIASES,
    SPECTRA_FIELDS,
    SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE,
    SPECTRA_SUMMARY_ALIASES,
    build_spectra_summary,
    empty_spectra_metrics,
    finite_positive_mask,
    load_spectra_metrics,
    relative_l2,
    relative_l2_weighted,
    spectra_field_dir_candidates,
    spectra_field_root,
    spectra_score,
    spectra_summary_keys,
)
from eval._backends.scoreboard.spectra import (
    _rescore_from_curve_summary,
)
from eval._backends.scoreboard.surface import (
    SURFACE_NORMALIZATION_SCHEME,
    SURFACE_VAR_LABELS,
    format_surface_loss_for_scoreboard,
    load_surface_loss_metrics,
    load_surface_weighted_mse,
    load_x_interp_surface_metrics,
    surface_weighted_nmse,
)
from eval._backends.scoreboard.tc import (
    MSLP_REFERENCE_HPA,
    load_tc_extreme_scores_from_json as _canonical_load_tc_extreme_scores_from_json,
    mslp_depth as _mslp_depth,
    multi_depth_enfo_deviation as _multi_depth_enfo_deviation,
    multi_depth_tc_score as _multi_depth_tc_score,
    normalize_tc_rows as _normalize_tc_rows,
)

# Canonical analysis: loaded from YAML for backward compat
CANONICAL_OPER_O320_ANALYSIS = _load_canonical_analysis("o320")

def load_tc_extreme_scores_from_json(
    stats_path: Path,
    *,
    run_id: str,
    event_names: tuple[str, ...] | list[str] | None = None,
    canonical_analysis_by_event: dict[str, dict[str, Any]] | None = None,
    canonical_eefo_by_event: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    requested = tuple(event_names or ("idalia", "franklin"))
    result = _canonical_load_tc_extreme_scores_from_json(
        stats_path,
        run_id=run_id,
        event_names=requested,
        canonical_analysis_by_event=canonical_analysis_by_event,
        canonical_eefo_by_event=canonical_eefo_by_event,
    )
    missing = [event for event in requested if event not in result]
    if missing and canonical_analysis_by_event is None and canonical_eefo_by_event is None:
        fallback = _canonical_load_tc_extreme_scores_from_json(
            stats_path,
            run_id=run_id,
            event_names=missing,
            canonical_analysis_by_event={},
            canonical_eefo_by_event={},
        )
        result.update(fallback)
    return result



# ---------------------------------------------------------------------------
# Constants and utilities that remain in this module (not extracted)
# ---------------------------------------------------------------------------

SIGMA_LEVELS = (1.0, 5.0, 10.0, 100.0)
CHECKPOINT_TOKEN_RE = re.compile(r"(?:^|manual_)([0-9a-f]{7,64})(?:_|$)")


def sigma_fragment(sigma: float) -> str:
    return f"{sigma:g}"


def load_mapping_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        try:
            data = yaml.safe_load(text)
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return dict(raw)
    if raw in (None, ""):
        return None
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extra_args_from_development_hacks(config: dict[str, Any]) -> dict[str, Any] | None:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    hacks = model.get("development_hacks")
    if not isinstance(hacks, dict):
        return None
    extra_args = hacks.get("extra_args")
    return dict(extra_args) if isinstance(extra_args, dict) else None


def parse_sampling_text_map(sampling_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(sampling_text or "").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def normalize_schedule_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text or text == "na":
        return ""
    if "experimental_piecewise" in text:
        return "piecewise"
    piecewise_match = re.search(r"(piecewise\d+)", text)
    if piecewise_match:
        return piecewise_match.group(1)
    if "piecewise" in text:
        return "piecewise"
    if "karras" in text:
        return "karras"
    if "exponential" in text:
        return "exponential"
    if "lognormal" in text or "lognorm" in text:
        return "lognorm"
    head = re.split(r"[_\s]", text, maxsplit=1)[0]
    return head if head and head != "experimental" else text


def format_step_count(raw: Any) -> str:
    if raw in (None, ""):
        return ""
    text = str(raw).strip()
    if not text or text.lower() == "na":
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not math.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def infer_schedule_label(values: dict[str, Any]) -> str:
    for key in ("schedule", "schedule_name", "schedule_type", "scheduler", "scheduler_type", "noise_schedule"):
        label = normalize_schedule_label(values.get(key))
        if label:
            return label
    high = normalize_schedule_label(values.get("high_schedule_type"))
    low = normalize_schedule_label(values.get("low_schedule_type"))
    if high or low:
        return "piecewise"
    has_step_count = any(format_step_count(values.get(key)) for key in ("num_steps", "steps", "n_steps"))
    if has_step_count and any(values.get(key) not in (None, "", "na") for key in ("rho", "sigma_max", "sigma_min")):
        return "karras"
    return ""


def parse_python_dict(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_extra_args_from_log(log_path: Path) -> dict[str, Any] | None:
    scheduler: dict[str, Any] | None = None
    sampler: dict[str, Any] | None = None
    try:
        with log_path.open(errors="ignore") as handle:
            for line in handle:
                if scheduler is None:
                    if "noise_scheduler_config:" in line:
                        scheduler = parse_python_dict(line.split("noise_scheduler_config:", 1)[1])
                    elif "noise_scheduler_params:" in line:
                        scheduler = parse_python_dict(line.split("noise_scheduler_params:", 1)[1])
                if sampler is None:
                    if "diffusion_sampler_config:" in line:
                        sampler = parse_python_dict(line.split("diffusion_sampler_config:", 1)[1])
                    elif "sampler_params:" in line:
                        sampler = parse_python_dict(line.split("sampler_params:", 1)[1])
                if scheduler is not None and sampler is not None:
                    break
    except OSError:
        return None

    if scheduler is None and sampler is None:
        return None
    merged: dict[str, Any] = {}
    if scheduler is not None:
        merged.update(scheduler)
    if sampler is not None:
        merged.update(sampler)
    return merged


def infer_eval_sampler_min(extra_args: dict[str, Any] | None, sampling_text: str = "") -> str:
    parsed_text = parse_sampling_text_map(sampling_text)
    merged: dict[str, Any] = dict(parsed_text)
    if isinstance(extra_args, dict):
        merged.update(extra_args)

    schedule_label = infer_schedule_label(merged)
    step_count = ""
    for key in ("num_steps", "steps", "n_steps"):
        step_count = format_step_count(merged.get(key))
        if step_count:
            break

    if schedule_label.startswith("piecewise") and re.search(r"\d+$", schedule_label):
        return schedule_label
    if schedule_label and step_count:
        return f"{schedule_label}{step_count}"
    if schedule_label:
        return schedule_label
    if step_count:
        return f"steps{step_count}"
    return "na"


def _infer_sampler_label_from_run_id(run_id: str) -> str:
    """Last-resort: extract sampler+steps from the run directory name."""
    match = re.search(r"(?:^|_)(heun|karras|piecewise|euler|dpm|edm|lognorm)(\d+)(?:_|$)", run_id.lower())
    if not match:
        return "na"
    schedule = normalize_schedule_label(match.group(1))
    steps = match.group(2)
    if schedule and steps:
        return f"{schedule}{steps}"
    return schedule or "na"


def infer_eval_sampler_min_from_run_root(run_root: Path) -> str:
    config_path = run_root / "EXPERIMENT_CONFIG.yaml"
    sampling_text = ""
    extra_args: dict[str, Any] | None = None

    if config_path.exists():
        config = load_mapping_file(config_path)
        parsed_sampling = parse_json_object(config.get("sampling_config_json"))
        if parsed_sampling is not None:
            extra_args = parsed_sampling
        else:
            raw_sampling = config.get("sampling_config_json")
            if raw_sampling not in (None, ""):
                sampling_text = str(raw_sampling).strip()
        if extra_args is None:
            config_extra_args = extra_args_from_development_hacks(config)
            if config_extra_args is not None:
                extra_args = config_extra_args

    label = infer_eval_sampler_min(extra_args, sampling_text)
    if label != "na":
        return label

    logs_dir = run_root / "logs"
    if not logs_dir.exists():
        return _infer_sampler_label_from_run_id(run_root.name)

    seen: set[Path] = set()
    log_candidates = (
        sorted(logs_dir.glob("predict25_*.out"))
        + sorted(logs_dir.glob("predict_proxy_*.out"))
        + sorted(logs_dir.glob("*.out"))
    )
    for log_path in log_candidates:
        if log_path in seen:
            continue
        seen.add(log_path)
        parsed_from_log = extract_extra_args_from_log(log_path)
        if parsed_from_log is None:
            continue
        label = infer_eval_sampler_min(parsed_from_log)
        if label != "na":
            return label

    return _infer_sampler_label_from_run_id(run_root.name)


def load_sigma_losses_from_csv(csv_path: Path) -> dict[str, float]:
    sigma_losses: dict[str, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sigma = finite_float(row.get("sigma"))
            loss = finite_float(row.get("loss"))
            if sigma is None or loss is None:
                continue
            sigma_losses[f"sigma_{sigma_fragment(sigma)}"] = loss
    return sigma_losses


def sigma_losses_for_scoreboard(csv_path: Path) -> dict[str, float | None]:
    sigma_losses = load_sigma_losses_from_csv(csv_path)
    return {
        f"sigma_{sigma_fragment(level)}": sigma_losses.get(f"sigma_{sigma_fragment(level)}")
        for level in SIGMA_LEVELS
    }


# ---------------------------------------------------------------------------
# Orchestration (stays here — not part of the scoring library)
# ---------------------------------------------------------------------------

def build_run_scoreboard_metrics(
    *,
    run_id: str,
    output_root: Path,
    sigma_run_id: str,
    tc_stats_path: Path,
    spectra_dir: Path,
    surface_json_path: Path,
    event_names: tuple[str, ...] | list[str] | None = None,
    lane: str = "",
    checkpoint_id: str = "",
    checkpoint_step: int = 0,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical metrics.json content for a run root.

    When lane/checkpoint_id/checkpoint_step are provided, the output follows
    the canonical eval-data-contract schema (schema_version 1.0). Otherwise
    falls back to legacy flat dict for backward compatibility.
    """
    import datetime

    canonical = bool(lane)

    sigma_losses = {
        f"sigma_{sigma_fragment(level)}": None
        for level in SIGMA_LEVELS
    }
    if sigma_run_id:
        sigma_csv = output_root / "scoreboards" / "sigma" / f"{sigma_run_id}_sigma_eval.csv"
        if sigma_csv.exists():
            sigma_losses.update(sigma_losses_for_scoreboard(sigma_csv))
    if all(v is None for v in sigma_losses.values()):
        # Try canonical location first, then legacy
        canonical_csv = output_root / run_id / "data" / "sigma_eval.csv"
        if canonical_csv.exists():
            sigma_losses.update(sigma_losses_for_scoreboard(canonical_csv))
        else:
            run_root_csv = output_root / run_id / "sigma_eval_table.csv"
            if run_root_csv.exists():
                sigma_losses.update(sigma_losses_for_scoreboard(run_root_csv))

    tc_scores = {}
    if tc_stats_path.is_dir():
        # Canonical: tc_stats_path is data/tc/ directory with per-event files
        for stats_file in sorted(tc_stats_path.glob("*.stats.json")):
            event_name = stats_file.stem.replace(".stats", "")
            event_scores = load_tc_extreme_scores_from_json(stats_file, run_id=run_id, event_names=(event_name,))
            tc_scores.update(event_scores)
    elif tc_stats_path.exists():
        tc_scores = load_tc_extreme_scores_from_json(tc_stats_path, run_id=run_id, event_names=event_names)

    spectra_metrics = load_spectra_metrics(spectra_dir)

    surface_metrics = load_surface_loss_metrics(surface_json_path)
    surface_loss = finite_float(surface_metrics.get("weighted_mse"))
    surface_nmse = finite_float(surface_metrics.get("weighted_nmse"))

    if not canonical:
        # Legacy output format
        out: dict[str, Any] = {"run_id": run_id}
        out["sigma_losses"] = sigma_losses
        out["tc_extreme_scores"] = tc_scores
        if spectra_metrics["mean"] is not None:
            out["spectra_mean_relative_l2"] = float(spectra_metrics["mean"])
            out["spectra_relative_l2"] = {
                field: float(spectra_metrics[field])
                for field in SPECTRA_FIELDS
                if spectra_metrics[field] is not None
            }
        if surface_loss is not None:
            out["surface_weighted_mse"] = surface_loss
        if surface_nmse is not None:
            out["surface_weighted_nmse"] = surface_nmse
        return out

    # Canonical schema_version 1.0
    scalars: dict[str, Any] = {}
    if spectra_metrics["mean"] is not None:
        scalars["spectra_mean_relative_l2"] = float(spectra_metrics["mean"])
    if surface_loss is not None:
        scalars["surface_weighted_mse"] = surface_loss
    if surface_nmse is not None:
        scalars["surface_weighted_nmse"] = surface_nmse
    for key, val in sigma_losses.items():
        if val is not None:
            scalars[key] = val

    # TC scores in canonical format
    tc_canonical: dict[str, Any] = {}
    for event_key, event_val in tc_scores.items():
        if isinstance(event_val, dict):
            tc_canonical[event_key] = event_val
        else:
            tc_canonical[event_key] = {"extreme_score": event_val}

    # Artifacts (relative paths)
    artifacts: dict[str, Any] = {
        "surface_loss": "data/surface_loss.json",
        "spectra_summary": "data/spectra/summary.json",
        "spectra_curves": "data/spectra/curves.json",
        "sigma_eval": "data/sigma_eval.csv",
    }
    if tc_scores:
        artifacts["tc_stats"] = {
            event: f"data/tc/{event}.stats.json" for event in tc_scores
        }

    out_canonical: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "lane": lane,
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "eval_date": datetime.date.today().isoformat(),
        "scope": scope or {},
        "scalars": scalars,
        "tc_scores": tc_canonical,
        "artifacts": artifacts,
    }
    return out_canonical
