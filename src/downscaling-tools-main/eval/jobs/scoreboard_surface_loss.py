#!/usr/bin/env python3
"""Compute area-weighted and variable-weighted surface MSE for scoreboard evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval._backends.scoreboard._surface_compute import (
    SURFACE_NORMALIZATION_SCHEME,
    SURFACE_VARIABLES,
    TOTAL_WEIGHT,
    process_predictions_dir,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute area-weighted and variable-weighted surface MSE for scoreboard"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing predictions_*.nc files",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        required=True,
        help="Output JSON file path (e.g., surface_loss_summary.json)",
    )
    parser.add_argument(
        "--prediction-var",
        default="y_pred",
        help="Dataset variable to score against truth (default: y_pred).",
    )
    parser.add_argument(
        "--truth-var",
        default="y",
        help="Dataset truth variable (default: y).",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    out_json = Path(args.out_json)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    print(f"Computing surface loss from: {predictions_dir}")
    print(f"Prediction variable: {args.prediction_var} | truth variable: {args.truth_var}")
    results = process_predictions_dir(
        predictions_dir,
        prediction_var=args.prediction_var,
        truth_var=args.truth_var,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\nResults written to: {out_json}")
    print(f"Weighted surface MSE: {results['weighted_surface_mse']:.6e}")
    weighted_surface_nmse = results.get("weighted_surface_nmse")
    if weighted_surface_nmse is None:
        print("Weighted surface nMSE: na")
    else:
        print(f"Weighted surface nMSE: {weighted_surface_nmse:.6e}")
    print("\nPer-variable breakdown:")
    for var, data in sorted(results["variables"].items()):
        mean_nmse_text = "na"
        if "mean_nmse" in data:
            mean_nmse_text = f"{data['mean_nmse']:.6e}"
        print(
            f"  {var:6s}: MSE={data['mean_mse']:.6e}, "
            f"nMSE={mean_nmse_text}, "
            f"weight={data['normalized_weight']:.3f}, "
            f"samples={data['n_member_samples']}"
        )


if __name__ == "__main__":
    main()
