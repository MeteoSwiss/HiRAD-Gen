"""TC (tropical cyclone) extreme scoring — analysis-anchored multi-depth scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._utils import finite_float as _finite_float
from eval._backends.scoreboard.row_matching import (
    classify_row,
    find_model_row,
    find_row_by_predicate,
    is_analysis_row,
    is_eefo_row,
    is_reference_row,
)
from eval.scoreboard.types import RowClassification

MSLP_REFERENCE_HPA = 1013.25
_OVERSHOOT_BETA = 0.5


def _asymmetric_ratio_score(ratio: float) -> float:
    """Score a model/analysis ratio with asymmetric penalty: overshoot penalized at half rate."""
    if ratio <= 1.0:
        return ratio
    return max(0.0, 1.0 - _OVERSHOOT_BETA * (ratio - 1.0))



def mslp_depth(value: float) -> float:
    """Convert MSLP (hPa) to depth below standard reference: deeper = more extreme."""
    return max(MSLP_REFERENCE_HPA - value, 0.0)


def multi_depth_tc_score(
    model: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute analysis-anchored TC score using multi-depth tail percentiles.

    Returns 0–1 score: 1.0 = model matches analysis extremes, 0.0 = no extreme signal.
    """
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    mslp_ratios: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        a_depth = mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        m_depth = mslp_depth(m_val)
        mslp_ratios.append(_asymmetric_ratio_score(m_depth / a_depth))

    wind_ratios: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_ratios.append(_asymmetric_ratio_score(m_val / a_val))

    if not mslp_ratios and not wind_ratios:
        return None

    scores: list[float] = []
    if mslp_ratios:
        scores.append(sum(mslp_ratios) / len(mslp_ratios))
    if wind_ratios:
        scores.append(sum(wind_ratios) / len(wind_ratios))
    return sum(scores) / len(scores)


def multi_depth_enfo_deviation(
    model: dict[str, Any],
    enfo: dict[str, Any],
    analysis: dict[str, Any],
) -> float | None:
    """Compute ENFO deviation: how far the model diverges from ENFO, normalized by analysis."""
    mslp_keys = ("mslp_p1", "mslp_p01", "mslp_min")
    wind_keys = ("wind_p99", "wind_p999", "wind_max")

    mslp_devs: list[float] = []
    for key in mslp_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        a_depth = mslp_depth(a_val)
        if a_depth <= 0.0:
            continue
        mslp_devs.append(abs(mslp_depth(m_val) - mslp_depth(e_val)) / a_depth)

    wind_devs: list[float] = []
    for key in wind_keys:
        m_val = _finite_float(model.get(key))
        e_val = _finite_float(enfo.get(key))
        a_val = _finite_float(analysis.get(key))
        if m_val is None or e_val is None or a_val is None:
            continue
        if a_val <= 0.0:
            continue
        wind_devs.append(abs(m_val - e_val) / a_val)

    if not mslp_devs and not wind_devs:
        return None

    devs: list[float] = []
    if mslp_devs:
        devs.append(sum(mslp_devs) / len(mslp_devs))
    if wind_devs:
        devs.append(sum(wind_devs) / len(wind_devs))
    return sum(devs) / len(devs)


def rescale_with_eefo_floor(raw_score: float, eefo_raw: float | None) -> float:
    """Rescale a raw TC score so EEFO maps to 0 and analysis maps to 1."""
    if eefo_raw is not None and eefo_raw < 1.0:
        return max(0.0, (raw_score - eefo_raw) / (1.0 - eefo_raw))
    return raw_score


def normalize_tc_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_analysis: dict[str, Any] | None = None,
    canonical_eefo: dict[str, Any] | None = None,
    event_name: str | None = None,
) -> None:
    """Analysis-anchored TC scoring with multi-depth tail percentiles.

    Parameters
    ----------
    rows : list of row dicts from TC stats JSON
    canonical_analysis : dict, optional
        Explicit canonical analysis values for this event. When provided and the
        embedded analysis is O320-class, overrides the embedded analysis row.
    event_name : str, optional
        Used only for backward compat with the old interface that looked up
        canonical values internally. Prefer passing canonical_analysis directly.

    Scores are rescaled so EEFO maps to 0 and analysis maps to 1.
    Falls back to legacy batch-relative normalization when analysis row or
    tail percentiles are not available.
    """
    embedded_analysis = find_row_by_predicate(rows, is_analysis_row)

    # Use canonical only when the embedded analysis is O320-class (not O1280)
    analysis_row: dict[str, Any] | None = None
    if canonical_analysis is not None:
        embedded_exp = str(embedded_analysis.get("exp", "")).upper() if embedded_analysis else ""
        if "O1280" not in embedded_exp:
            analysis_row = canonical_analysis

    if analysis_row is None:
        analysis_row = embedded_analysis if isinstance(embedded_analysis, dict) else None

    enfo_row = find_row_by_predicate(rows, is_reference_row)

    if analysis_row is not None and _finite_float(analysis_row.get("mslp_p1")) is not None:
        eefo_raw = None
        if canonical_eefo is not None:
            eefo_raw = multi_depth_tc_score(canonical_eefo, analysis_row)
        else:
            eefo_row = find_row_by_predicate(rows, is_eefo_row)
            if eefo_row is not None:
                eefo_raw = multi_depth_tc_score(eefo_row, analysis_row)

        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if is_analysis_row(exp):
                row["_extreme_score_value"] = 1.0
                row["_enfo_deviation_value"] = None
                continue
            raw_score = multi_depth_tc_score(row, analysis_row)
            if raw_score is not None:
                row["_extreme_score_value"] = rescale_with_eefo_floor(raw_score, eefo_raw)
            else:
                row["_extreme_score_value"] = None
            if enfo_row is not None:
                row["_enfo_deviation_value"] = multi_depth_enfo_deviation(row, enfo_row, analysis_row)
            else:
                row["_enfo_deviation_value"] = None
        return

    # Legacy fallback: batch-relative normalization
    m_values = [v for v in (_finite_float(row.get("mslp_980_990_fraction")) for row in rows) if v is not None]
    w_values = [v for v in (_finite_float(row.get("wind_gt_25_fraction")) for row in rows) if v is not None]
    m_min = min(m_values) if m_values else None
    m_max = max(m_values) if m_values else None
    w_min = min(w_values) if w_values else None
    w_max = max(w_values) if w_values else None

    for row in rows:
        score = _finite_float(row.get("extreme_score"))
        if score is not None:
            row["_extreme_score_value"] = score
            row["_enfo_deviation_value"] = None
            continue

        m_val = _finite_float(row.get("mslp_980_990_fraction"))
        w_val = _finite_float(row.get("wind_gt_25_fraction"))
        if m_val is None or w_val is None or m_min is None or m_max is None or w_min is None or w_max is None:
            row["_extreme_score_value"] = None
            row["_enfo_deviation_value"] = None
            continue

        m_norm = 0.0 if m_max <= m_min else (m_val - m_min) / (m_max - m_min)
        w_norm = 0.0 if w_max <= w_min else (w_val - w_min) / (w_max - w_min)
        row["_extreme_score_value"] = 0.5 * (m_norm + w_norm)
        row["_enfo_deviation_value"] = None


def load_tc_extreme_scores_from_json(
    stats_path: Path,
    *,
    run_id: str,
    event_names: tuple[str, ...] | list[str] | None = None,
    canonical_analysis_by_event: dict[str, dict[str, Any]] | None = None,
    canonical_eefo_by_event: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Load TC extreme scores and ENFO deviation from a stats JSON.

    Parameters
    ----------
    stats_path : Path to the TC stats JSON file.
    run_id : The experiment run ID to match.
    event_names : Which events to score. Defaults to ("idalia", "franklin").
    canonical_analysis_by_event : dict mapping event name -> canonical analysis dict.
        When provided, used instead of internal lookup. Pass None to use the
        old behavior of looking up CANONICAL_OPER_O320_ANALYSIS by event name.
    canonical_eefo_by_event : dict mapping event name -> canonical EEFO dict.
        When provided, used as the EEFO floor instead of finding an EEFO row.

    Returns
    -------
    dict with keys like "idalia", "franklin" (scores) and
    "idalia_enfo_dev", "franklin_enfo_dev" (deviations, when available).
    """
    with stats_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
    events = data.get("events")
    if not isinstance(events, dict):
        return {}

    # Backward compat: if no canonical provided, load from YAML
    if canonical_analysis_by_event is None:
        from eval._backends.scoreboard.canonical_data import load_canonical_analysis

        canonical_analysis_by_event = load_canonical_analysis("o320")

    requested_events = tuple(event_names or ("idalia", "franklin"))
    scores: dict[str, float] = {}
    for event_name in requested_events:
        event_data = events.get(event_name)
        if not isinstance(event_data, dict):
            continue
        rows = event_data.get("extreme_tail", {}).get("rows", [])
        if not isinstance(rows, list):
            rows = event_data.get("rows", [])
        if not isinstance(rows, list):
            continue
        norm_rows = [row for row in rows if isinstance(row, dict)]

        canonical = canonical_analysis_by_event.get(event_name) if canonical_analysis_by_event else None
        eefo = canonical_eefo_by_event.get(event_name) if canonical_eefo_by_event else None
        normalize_tc_rows(norm_rows, canonical_analysis=canonical, canonical_eefo=eefo, event_name=event_name)

        chosen = find_model_row(norm_rows, run_id)
        if chosen is None:
            continue
        score = _finite_float(chosen.get("_extreme_score_value"))
        if score is not None:
            scores[event_name] = score
        enfo_dev = _finite_float(chosen.get("_enfo_deviation_value"))
        if enfo_dev is not None:
            scores[f"{event_name}_enfo_dev"] = enfo_dev
    return scores
