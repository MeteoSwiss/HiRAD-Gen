"""Pure statistics for TC evaluation — no matplotlib."""
from __future__ import annotations

import math

import numpy as np


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def summary_stats(x: np.ndarray) -> dict:
    vals = _finite_1d(x)
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q01": float(np.quantile(vals, 0.01)),
        "q05": float(np.quantile(vals, 0.05)),
        "q50": float(np.quantile(vals, 0.50)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def tail_summary(x: np.ndarray, *, tail: str) -> dict:
    x = _finite_1d(x)
    if x.size == 0:
        return {"n": 0}

    p = np.percentile(x, [0.1, 1, 5, 50, 95, 99, 99.5, 99.9])
    p01, p1, p5, p50, p95, p99, p995, p999 = p
    out = {
        "n": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "p0.1": float(p01),
        "p1": float(p1),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "p99.5": float(p995),
        "p99.9": float(p999),
    }

    eps = 1e-12
    if tail == "high":
        denom = max(p95 - p50, eps)
        out["tail_index"] = float((p99 - p95) / denom)
        out["extreme_index"] = float((p999 - p99) / denom)
        out["top0.1_mean"] = float(x[x >= p999].mean()) if np.any(x >= p999) else np.nan
    elif tail == "low":
        denom = max(p50 - p5, eps)
        out["tail_index"] = float((p5 - p1) / denom)
        out["extreme_index"] = float((p1 - p01) / denom)
        out["bottom0.1_mean"] = float(x[x <= p01].mean()) if np.any(x <= p01) else np.nan
    else:
        raise ValueError("tail must be 'low' or 'high'")
    return out


def distribution_metrics(hist_ref: np.ndarray, hist_other: np.ndarray, bin_width: float) -> dict:
    pref = np.asarray(hist_ref, dtype=np.float64) * bin_width
    poth = np.asarray(hist_other, dtype=np.float64) * bin_width
    sref = float(np.sum(pref))
    soth = float(np.sum(poth))
    if sref <= 0.0 or soth <= 0.0:
        return {
            "l1_mass": math.nan,
            "total_variation": math.nan,
            "rmse_density": math.nan,
            "max_abs_density_diff": math.nan,
            "ks_hist": math.nan,
            "kl_ref_to_other": math.nan,
            "kl_other_to_ref": math.nan,
            "js_divergence": math.nan,
        }
    pref /= sref
    poth /= soth
    diff = pref - poth
    l1 = float(np.sum(np.abs(diff)))
    tv = 0.5 * l1
    max_abs_density_diff = float(np.max(np.abs(hist_ref - hist_other)))
    rmse_density = float(np.sqrt(np.mean((hist_ref - hist_other) ** 2)))
    cdf_ref = np.cumsum(pref)
    cdf_oth = np.cumsum(poth)
    ks_hist = float(np.max(np.abs(cdf_ref - cdf_oth)))
    eps = 1e-12
    kl_ref_to_other = float(np.sum(pref * np.log((pref + eps) / (poth + eps))))
    kl_other_to_ref = float(np.sum(poth * np.log((poth + eps) / (pref + eps))))
    mean_prob = 0.5 * (pref + poth)
    js = 0.5 * (
        np.sum(pref * np.log((pref + eps) / (mean_prob + eps)))
        + np.sum(poth * np.log((poth + eps) / (mean_prob + eps)))
    )
    return {
        "l1_mass": l1,
        "total_variation": tv,
        "rmse_density": rmse_density,
        "max_abs_density_diff": max_abs_density_diff,
        "ks_hist": ks_hist,
        "kl_ref_to_other": kl_ref_to_other,
        "kl_other_to_ref": kl_other_to_ref,
        "js_divergence": float(js),
    }


def ratio_metrics(hist_ref: np.ndarray, hist_other: np.ndarray) -> dict:
    ratio = safe_ratio(hist_other, hist_ref)
    valid = np.isfinite(ratio)
    if not np.any(valid):
        return {
            "valid_bins": 0,
            "ratio_mean": math.nan,
            "ratio_std": math.nan,
            "ratio_min": math.nan,
            "ratio_max": math.nan,
            "ratio_mae_to_1": math.nan,
            "ratio_max_abs_dev_from_1": math.nan,
        }
    values = ratio[valid]
    return {
        "valid_bins": int(values.size),
        "ratio_mean": float(np.mean(values)),
        "ratio_std": float(np.std(values)),
        "ratio_min": float(np.min(values)),
        "ratio_max": float(np.max(values)),
        "ratio_mae_to_1": float(np.mean(np.abs(values - 1.0))),
        "ratio_max_abs_dev_from_1": float(np.max(np.abs(values - 1.0))),
    }


def extreme_fraction_mslp(vals: np.ndarray, mslp_range: tuple[float, float]) -> float:
    vals = _finite_1d(vals)
    if vals.size == 0:
        return math.nan
    lo, hi = mslp_range
    return float(np.mean((vals >= lo) & (vals <= hi)))


def extreme_fraction_wind(vals: np.ndarray, wind_gt: float) -> float:
    vals = _finite_1d(vals)
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals > wind_gt))


def variable_stats(
    vals: np.ndarray,
    *,
    hist_ref: np.ndarray,
    bins: np.ndarray,
    bin_width: float,
    tail: str,
) -> tuple[np.ndarray, dict]:
    hist, _ = np.histogram(vals, bins=bins, density=True)
    stats = {
        "summary": summary_stats(vals),
        "tail": tail_summary(vals, tail=tail),
        "vs_oper": {
            **distribution_metrics(hist_ref, hist, bin_width),
            **ratio_metrics(hist_ref, hist),
        },
    }
    return hist, stats


def extreme_tail_table(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    mslp_range: tuple[float, float] = (980.0, 990.0),
    wind_threshold: float = 25.0,
) -> dict:
    """Compute extreme tail metrics for multiple experiments.

    Thresholds are explicit parameters — the caller decides per event.
    """
    rows: list[dict[str, object]] = []
    for exp, (msl_arr, wind_arr) in series.items():
        msl = _finite_1d(msl_arr)
        wind = _finite_1d(wind_arr)
        msl_hit = (msl >= mslp_range[0]) & (msl <= mslp_range[1])
        wind_hit = wind > wind_threshold
        row: dict[str, object] = {
            "exp": exp,
            "mslp_980_990_count": int(np.sum(msl_hit)),
            "mslp_980_990_fraction": float(np.mean(msl_hit)) if msl.size else math.nan,
            "wind_gt_25_count": int(np.sum(wind_hit)),
            "wind_gt_25_fraction": float(np.mean(wind_hit)) if wind.size else math.nan,
            "n_msl": int(msl.size),
            "n_wind": int(wind.size),
        }
        if msl.size > 0:
            row["mslp_p1"] = float(np.percentile(msl, 1.0))
            row["mslp_p01"] = float(np.percentile(msl, 0.1))
            row["mslp_min"] = float(np.min(msl))
        if wind.size > 0:
            row["wind_p99"] = float(np.percentile(wind, 99.0))
            row["wind_p999"] = float(np.percentile(wind, 99.9))
            row["wind_max"] = float(np.max(wind))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["mslp_980_990_fraction"]) if np.isfinite(row["mslp_980_990_fraction"]) else -1.0,
            float(row["wind_gt_25_fraction"]) if np.isfinite(row["wind_gt_25_fraction"]) else -1.0,
        ),
        reverse=True,
    )
    return {
        "thresholds": {
            "mslp_hpa_range": [mslp_range[0], mslp_range[1]],
            "wind_ms_gt": wind_threshold,
        },
        "rows": rows,
    }
