#!/usr/bin/env python3
"""Validate an overfit test: predictions sanity + spectra quality.

Usage (single variable):
    python validate_overfit.py \
        --predictions-nc /path/to/predictions.nc \
        --spectra-json /path/to/spectra_summary.json \
        --variable 2t

Usage (multi-variable):
    python validate_overfit.py \
        --predictions-nc /path/to/predictions.nc \
        --spectra-json /path/to/spectra_summary.json \
        --variables 2t,tp

Per-variable thresholds are built-in. Override globally with CLI flags.
Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Per-variable default thresholds
VARIABLE_DEFAULTS = {
    "2t": {"range_min": 180.0, "range_max": 340.0, "rmse": 2.0, "corr": 0.95, "ff_l2": 0.05, "res_l2": 1.0, "unit": "K"},
    "tp": {"range_min": -0.01, "range_max": 0.5, "rmse": 0.01, "corr": 0.3, "ff_l2": 0.5, "res_l2": 2.0, "unit": "kg/m²"},
}
# Fallback for unknown variables
DEFAULT_THRESHOLDS = {"range_min": -1e6, "range_max": 1e6, "rmse": 10.0, "corr": 0.5, "ff_l2": 0.1, "res_l2": 2.0, "unit": ""}


def _get_thresholds(variable: str, args) -> dict:
    """Get thresholds for a variable, using CLI overrides if provided."""
    defaults = VARIABLE_DEFAULTS.get(variable, DEFAULT_THRESHOLDS)
    return {
        "range_min": args.range_min if args.range_min is not None else defaults["range_min"],
        "range_max": args.range_max if args.range_max is not None else defaults["range_max"],
        "rmse": args.rmse_threshold if args.rmse_threshold is not None else defaults["rmse"],
        "corr": args.corr_threshold if args.corr_threshold is not None else defaults["corr"],
        "ff_l2": args.full_field_threshold if args.full_field_threshold is not None else defaults["ff_l2"],
        "res_l2": args.residual_threshold if args.residual_threshold is not None else defaults["res_l2"],
        "unit": defaults.get("unit", ""),
    }


def check_predictions(nc_path: str, variable: str, thresholds: dict) -> dict:
    """Basic sanity checks on predictions.nc."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    ws_list = list(ds.coords.get("weather_state", ds.get("weather_state", [])).values)

    if variable not in ws_list:
        return {"pass": False, "reason": f"Variable '{variable}' not found in weather_states: {ws_list}"}

    var_idx = ws_list.index(variable)

    y_pred = ds["y_pred"].values  # [sample, member, grid, vars]
    y_truth = ds["y"].values

    pred_var = y_pred[..., var_idx]
    truth_var = y_truth[..., var_idx]

    nan_frac = np.isnan(pred_var).sum() / pred_var.size
    inf_frac = np.isinf(pred_var).sum() / pred_var.size
    vmin, vmax = float(np.nanmin(pred_var)), float(np.nanmax(pred_var))
    vmean = float(np.nanmean(pred_var))

    rmse = float(np.sqrt(np.nanmean((pred_var - truth_var) ** 2)))
    corr = float(np.corrcoef(pred_var.ravel(), truth_var.ravel())[0, 1])

    results = {
        "variable": variable,
        "nan_fraction": nan_frac,
        "inf_fraction": inf_frac,
        "pred_min": vmin,
        "pred_max": vmax,
        "pred_mean": vmean,
        "rmse": rmse,
        "spatial_correlation": corr,
        "checks": {},
    }

    results["checks"]["no_nan"] = nan_frac == 0.0
    results["checks"]["no_inf"] = inf_frac == 0.0
    results["checks"]["range_ok"] = vmin >= thresholds["range_min"] and vmax <= thresholds["range_max"]
    results["checks"]["rmse_ok"] = rmse < thresholds["rmse"]
    results["checks"]["corr_ok"] = corr > thresholds["corr"]

    results["pass"] = all(results["checks"].values())
    return results


def check_spectra(spectra_json_path: str, variable: str, ff_threshold: float, res_threshold: float) -> dict:
    """Check spectra summary for a variable."""
    with open(spectra_json_path) as f:
        summary = json.load(f)

    ws_data = summary.get("weather_states", {})
    if variable not in ws_data:
        return {"pass": False, "reason": f"Variable '{variable}' not in spectra summary. Available: {list(ws_data.keys())}"}

    var_data = ws_data[variable]
    scopes = var_data.get("scopes", {})

    ff_score = None
    res_score = None

    if "full_field" in scopes:
        ff_score = scopes["full_field"].get("relative_l2_mean_curve")
    if "residual" in scopes:
        res_score = scopes["residual"].get("relative_l2_mean_curve")

    results = {
        "variable": variable,
        "full_field_relative_l2": ff_score,
        "residual_relative_l2": res_score,
        "checks": {},
    }

    if ff_score is not None and not (isinstance(ff_score, float) and np.isnan(ff_score)):
        results["checks"]["full_field_ok"] = ff_score < ff_threshold
    else:
        results["checks"]["full_field_ok"] = False
        results["full_field_note"] = "NaN or missing — likely lmax below score_wavenumber_min_exclusive"

    if res_score is not None and not (isinstance(res_score, float) and np.isnan(res_score)):
        results["checks"]["residual_ok"] = res_score < res_threshold
    else:
        results["checks"]["residual_ok"] = False
        results["residual_note"] = "NaN or missing"

    results["pass"] = all(results["checks"].values())
    return results


def main():
    p = argparse.ArgumentParser(description="Validate overfit test predictions + spectra.")
    p.add_argument("--predictions-nc", required=True, help="Path to predictions.nc")
    p.add_argument("--spectra-json", required=True, help="Path to spectra_summary.json")
    # Accept either --variable (single) or --variables (comma-separated list)
    p.add_argument("--variable", default=None, help="Single variable to validate (legacy)")
    p.add_argument("--variables", default=None, help="Comma-separated list of variables (e.g. 2t,tp)")
    # Optional global overrides (None = use per-variable defaults)
    p.add_argument("--full-field-threshold", type=float, default=None)
    p.add_argument("--residual-threshold", type=float, default=None)
    p.add_argument("--rmse-threshold", type=float, default=None)
    p.add_argument("--corr-threshold", type=float, default=None)
    p.add_argument("--range-min", type=float, default=None)
    p.add_argument("--range-max", type=float, default=None)
    args = p.parse_args()

    # Resolve variable list: --variables takes precedence, fall back to --variable
    if args.variables:
        var_list = [v.strip() for v in args.variables.split(",")]
    elif args.variable:
        var_list = [args.variable]
    else:
        var_list = ["2t"]

    print("=" * 60)
    print("OVERFIT VALIDATION TEST")
    print(f"Variables: {', '.join(var_list)}")
    print("=" * 60)

    all_pass = True

    for variable in var_list:
        thresholds = _get_thresholds(variable, args)
        unit = thresholds["unit"]

        # --- Predictions sanity ---
        print(f"\n--- Predictions sanity ({variable}) ---")
        pred_result = check_predictions(args.predictions_nc, variable, thresholds)
        if "reason" in pred_result:
            print(f"  FAIL: {pred_result['reason']}")
        else:
            print(f"  NaN fraction:  {pred_result['nan_fraction']:.6f}  {'✓' if pred_result['checks']['no_nan'] else '✗'}")
            print(f"  Inf fraction:  {pred_result['inf_fraction']:.6f}  {'✓' if pred_result['checks']['no_inf'] else '✗'}")
            print(f"  Range:         [{pred_result['pred_min']:.4f}, {pred_result['pred_max']:.4f}]  {'✓' if pred_result['checks']['range_ok'] else '✗'} (expected [{thresholds['range_min']}, {thresholds['range_max']}])")
            print(f"  RMSE vs truth: {pred_result['rmse']:.4f} {unit}  {'✓' if pred_result['checks']['rmse_ok'] else '✗'} (threshold {thresholds['rmse']})")
            print(f"  Spatial corr:  {pred_result['spatial_correlation']:.6f}  {'✓' if pred_result['checks']['corr_ok'] else '✗'} (threshold {thresholds['corr']})")
        print(f"  Predictions: {'PASS' if pred_result['pass'] else 'FAIL'}")

        # --- Spectra quality ---
        print(f"\n--- Spectra quality ({variable}) ---")
        spec_result = check_spectra(
            args.spectra_json, variable,
            thresholds["ff_l2"], thresholds["res_l2"],
        )
        if "reason" in spec_result:
            print(f"  FAIL: {spec_result['reason']}")
        else:
            ff = spec_result["full_field_relative_l2"]
            res = spec_result["residual_relative_l2"]
            ff_str = f"{ff:.6f}" if ff is not None else "N/A"
            res_str = f"{res:.6f}" if res is not None else "N/A"
            print(f"  Full-field relative_l2: {ff_str}  {'✓' if spec_result['checks']['full_field_ok'] else '✗'} (threshold {thresholds['ff_l2']})")
            if "full_field_note" in spec_result:
                print(f"    Note: {spec_result['full_field_note']}")
            print(f"  Residual relative_l2:  {res_str}  {'✓' if spec_result['checks']['residual_ok'] else '✗'} (threshold {thresholds['res_l2']})")
            if "residual_note" in spec_result:
                print(f"    Note: {spec_result['residual_note']}")
        print(f"  Spectra: {'PASS' if spec_result['pass'] else 'FAIL'}")

        if not (pred_result["pass"] and spec_result["pass"]):
            all_pass = False

    # --- Overall verdict ---
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'} ({len(var_list)} variable(s): {', '.join(var_list)})")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
