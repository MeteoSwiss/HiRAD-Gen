#!/usr/bin/env python
"""
Per-bundle TC extreme diagnostic.

Analyzes existing prediction files and computes TC extreme metrics
(mslp_980_990_fraction, wind_gt_25_fraction) per date/step combination,
separately for each TC event (Idalia, Franklin).

This identifies which of the 25 date/step combos contribute most to TC
extreme signal, enabling selection of a small proxy subset (~10 bundles)
for fast checkpoint screening.

Usage:
    python -m eval.jobs.diagnose_per_bundle_tc_extremes \
        --predictions-dir /home/ecm5702/scratch/eval/<RUN_ID>/predictions \
        --out-json /home/ecm5702/scratch/per_bundle_tc_extremes.json \
        --support-mode regridded \
        --events idalia,franklin \
        --analysis-expid OPER_O320_0001
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np

from eval.paths import reference_tc_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tc.events import EVENTS, TCEvent
from tc.data_types import CurveVectors, SupportMode
from tc.experiment_config import EXPERIMENT_CONFIGS
from tc.plot_config import PLOT_CONFIGS
from tc.loading_predictions import (
    analysis_dates_for_event,
    discover_prediction_files,
    load_prediction_curves,
    select_prediction_files_for_event,
)
from tc.loading_grib import regridded_target_points

LOG = logging.getLogger(__name__)

MSLP_EXTREME_RANGE = (980.0, 990.0)
WIND_EXTREME_MIN = 25.0


def _extreme_metrics(curve: CurveVectors) -> dict:
    """Compute extreme tail metrics for a single curve."""
    msl = np.asarray(curve.msl, dtype=np.float64).ravel()
    wind = np.asarray(curve.wind, dtype=np.float64).ravel()
    msl = msl[np.isfinite(msl)]
    wind = wind[np.isfinite(wind)]

    msl_hit = (msl >= MSLP_EXTREME_RANGE[0]) & (msl <= MSLP_EXTREME_RANGE[1])
    wind_hit = wind > WIND_EXTREME_MIN

    result = {
        "mslp_980_990_count": int(np.sum(msl_hit)),
        "mslp_980_990_fraction": float(np.mean(msl_hit)) if msl.size else 0.0,
        "wind_gt_25_count": int(np.sum(wind_hit)),
        "wind_gt_25_fraction": float(np.mean(wind_hit)) if wind.size else 0.0,
        "n_msl": int(msl.size),
        "n_wind": int(wind.size),
    }

    # Tail percentiles for MSLP (low tail) and wind (high tail)
    if msl.size > 0:
        result["mslp_p01"] = float(np.percentile(msl, 0.1))
        result["mslp_p1"] = float(np.percentile(msl, 1.0))
        result["mslp_min"] = float(np.min(msl))
    if wind.size > 0:
        result["wind_p99"] = float(np.percentile(wind, 99.0))
        result["wind_p999"] = float(np.percentile(wind, 99.9))
        result["wind_max"] = float(np.max(wind))

    return result


def diagnose_per_bundle(
    predictions_dir: str,
    *,
    base_tc_dir: str = str(reference_tc_dir("o96_o320")),
    support_mode: SupportMode = "regridded",
    event_names: list[str] | None = None,
    analysis_expid: str | None = None,
) -> dict:
    """Compute per date/step TC extreme metrics for each event."""
    pred_dir = Path(predictions_dir).expanduser().resolve()
    all_pred_files = discover_prediction_files(pred_dir)
    if not all_pred_files:
        raise FileNotFoundError(f"No predictions_*.nc in {pred_dir}")

    selected_events = event_names or ["idalia", "franklin"]
    results: dict[str, object] = {
        "predictions_dir": str(pred_dir),
        "support_mode": support_mode,
        "thresholds": {
            "mslp_hpa_range": list(MSLP_EXTREME_RANGE),
            "wind_ms_gt": WIND_EXTREME_MIN,
        },
        "events": {},
    }

    for event_name in selected_events:
        cfg = EVENTS[event_name]
        event_pred_files = select_prediction_files_for_event(all_pred_files, cfg)
        if not event_pred_files:
            LOG.warning("No prediction files for event=%s, skipping", event_name)
            continue

        # Precompute regridded target grid once per event
        target_lon, target_lat = None, None
        exp_cfg = EXPERIMENT_CONFIGS.get(event_name)
        plot_cfg = PLOT_CONFIGS.get(event_name)
        if support_mode == "regridded":
            days = sorted({int(f"{ymd:08d}"[6:8]) for _, ymd, _ in event_pred_files})
            sample_date = analysis_dates_for_event(cfg, days)[0]
            event_analysis = analysis_expid or (exp_cfg.analysis_expid if exp_cfg else None)
            if event_analysis is None:
                raise ValueError(
                    f"analysis_expid required for regridded mode (event={event_name}). "
                    "Pass --analysis-expid explicitly."
                )
            regrid_res = plot_cfg.regrid_resolution if plot_cfg else 0.25
            sample_path = f"{base_tc_dir}/{event_name}/surface_an_{event_analysis}_{sample_date}.grib"
            target_lon, target_lat = regridded_target_points(
                cfg.bbox, regrid_res, sample_path,
            )

        per_bundle: list[dict] = []

        # Analyze each prediction file individually (one per date/step)
        for path, ymd, step in event_pred_files:
            date_str = f"{ymd:08d}"
            LOG.info("  event=%s date=%s step=%03d ...", event_name, date_str, step)

            single_file = [(path, ymd, step)]
            try:
                curve = load_prediction_curves(
                    single_file,
                    bbox=cfg.bbox,
                    support_mode=support_mode,
                    target_lon=target_lon,
                    target_lat=target_lat,
                )
            except Exception as exc:
                LOG.error("Failed to load %s: %s", path.name, exc)
                continue

            metrics = _extreme_metrics(curve)
            metrics["date"] = date_str
            metrics["step"] = step
            metrics["file"] = path.name
            per_bundle.append(metrics)
            del curve
            gc.collect()

        # Compute aggregate from per-bundle sums (avoids loading all files at once)
        agg_mslp_count = sum(r["mslp_980_990_count"] for r in per_bundle)
        agg_wind_count = sum(r["wind_gt_25_count"] for r in per_bundle)
        agg_n_msl = sum(r["n_msl"] for r in per_bundle)
        agg_n_wind = sum(r["n_wind"] for r in per_bundle)
        agg_metrics = {
            "mslp_980_990_count": agg_mslp_count,
            "mslp_980_990_fraction": agg_mslp_count / agg_n_msl if agg_n_msl else 0.0,
            "wind_gt_25_count": agg_wind_count,
            "wind_gt_25_fraction": agg_wind_count / agg_n_wind if agg_n_wind else 0.0,
            "n_msl": agg_n_msl,
            "n_wind": agg_n_wind,
        }

        # Sort by combined extreme signal (mslp fraction + wind fraction)
        per_bundle.sort(
            key=lambda r: r["mslp_980_990_fraction"] + r["wind_gt_25_fraction"],
            reverse=True,
        )

        # Add rank
        for rank, row in enumerate(per_bundle, 1):
            row["extreme_rank"] = rank

        results["events"][event_name] = {
            "n_files": len(event_pred_files),
            "per_bundle": per_bundle,
            "aggregate": agg_metrics,
        }

    # Compute combined ranking across events
    bundle_scores: dict[str, float] = {}
    for event_name, event_data in results["events"].items():
        for row in event_data["per_bundle"]:
            key = f"{row['date']}_step{row['step']:03d}"
            score = row["mslp_980_990_fraction"] + row["wind_gt_25_fraction"]
            bundle_scores[key] = bundle_scores.get(key, 0.0) + score

    ranked_combined = sorted(bundle_scores.items(), key=lambda x: x[1], reverse=True)
    results["combined_ranking"] = [
        {"bundle": k, "combined_extreme_score": v, "rank": i + 1}
        for i, (k, v) in enumerate(ranked_combined)
    ]

    # Top 10 recommendation
    top10 = ranked_combined[:10]
    results["recommended_proxy_bundles"] = {
        "count": len(top10),
        "bundles": [k for k, _ in top10],
        "rationale": (
            "Top 10 date/step combos by combined mslp_980_990_fraction + wind_gt_25_fraction "
            "across Idalia and Franklin. Each bundle includes all 10 ensemble members."
        ),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose per-bundle TC extreme contributions."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base-tc-dir", default=str(reference_tc_dir("o96_o320")))
    parser.add_argument(
        "--support-mode",
        choices=["native", "regridded"],
        default="regridded",
    )
    parser.add_argument(
        "--events",
        default="idalia,franklin",
        help="Comma-separated event names.",
    )
    parser.add_argument(
        "--analysis-expid",
        default="",
        help="Analysis truth expid required for regridded mode, e.g. OPER_O320_0001.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    event_names = [e.strip() for e in args.events.split(",") if e.strip()]

    results = diagnose_per_bundle(
        args.predictions_dir,
        base_tc_dir=args.base_tc_dir,
        support_mode=args.support_mode,
        event_names=event_names,
        analysis_expid=args.analysis_expid or None,
    )

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOG.info("Wrote per-bundle TC extreme diagnostics to %s", out_path)

    # Print summary
    print("\n=== COMBINED RANKING (all events) ===")
    for entry in results["combined_ranking"]:
        marker = " <-- TOP 10" if entry["rank"] <= 10 else ""
        print(f"  #{entry['rank']:2d}  {entry['bundle']}  score={entry['combined_extreme_score']:.6e}{marker}")

    print(f"\n=== RECOMMENDED PROXY BUNDLES ({results['recommended_proxy_bundles']['count']}) ===")
    for b in results["recommended_proxy_bundles"]["bundles"]:
        print(f"  {b}")


if __name__ == "__main__":
    main()
