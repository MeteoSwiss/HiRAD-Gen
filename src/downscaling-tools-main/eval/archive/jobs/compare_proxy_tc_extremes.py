#!/usr/bin/env python
"""
DEPRECATED (2026-05-07): the `_compute_metrics_from_predictions` path imports
`eval.tc_extremes`, which does not exist in the live tree. Only the
`--anchor-ref` path is functional. Do not extend; use
`python -m eval.evaluators.tc.runner` (via `eval.cli evaluate --only tc`) for
new TC scoring needs.
"""
"""
Compare proxy TC extremes against the anchor baseline.

After running proxy predictions (10 bundles), this script computes TC extreme
metrics and compares them to the anchor reference, giving a quick read on
whether the checkpoint improves or degrades TC representation.

Usage:
    python -m eval.jobs.compare_proxy_tc_extremes \
        --predictions-dir /path/to/new_run/predictions \
        --anchor-ref /home/ecm5702/scratch/per_bundle_tc_extremes_56b6c4e2_anchor.json \
        --out /home/ecm5702/scratch/proxy_comparison_<run_id>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

# Proxy bundle pairs (date, step) — top 10 by combined TC extreme signal
PROXY_BUNDLES = [
    ("20230829", 24),
    ("20230828", 48),
    ("20230829", 48),
    ("20230828", 24),
    ("20230830", 24),
    ("20230828", 72),
    ("20230827", 72),
    ("20230830", 48),
    ("20230829", 72),
    ("20230827", 48),
]


def _load_anchor_proxy_metrics(anchor_path: Path) -> dict:
    """Extract only the proxy bundle metrics from the full anchor diagnostic."""
    with open(anchor_path) as f:
        anchor = json.load(f)

    proxy_keys = {f"{d}_step{s:03d}" for d, s in PROXY_BUNDLES}
    result = {}
    for event_name, event_data in anchor["events"].items():
        bundles = {}
        for b in event_data["per_bundle"]:
            key = f"{b['date']}_step{b['step']:03d}"
            if key in proxy_keys:
                bundles[key] = b
        result[event_name] = bundles
    return result


def _compute_metrics_from_predictions(predictions_dir: Path, support_mode: str) -> dict:
    """Compute per-bundle TC extreme metrics for all prediction files in the directory."""
    # Import here to avoid import cost when just loading anchor
    from eval.tc_extremes import TC_EVENTS, load_prediction_event_curve

    from eval.jobs.diagnose_per_bundle_tc_extremes import _extreme_metrics

    import gc
    import re

    pred_re = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
    pred_files = sorted(predictions_dir.glob("predictions_*.nc"))

    if not pred_files:
        raise SystemExit(f"No prediction files found in {predictions_dir}")

    LOG.info("Found %d prediction files in %s", len(pred_files), predictions_dir)

    cfg = {"support_mode": support_mode}
    result = {}

    for event_name, event_cfg in TC_EVENTS.items():
        target_lon = event_cfg["target_lon"]
        target_lat = event_cfg["target_lat"]
        bundles = {}

        for path in pred_files:
            m = pred_re.search(path.name)
            if not m:
                continue
            date, step = m.group(1), int(m.group(2))
            key = f"{date}_step{step:03d}"

            LOG.info("  event=%s %s ...", event_name, key)
            try:
                curve = load_prediction_event_curve(
                    [path], cfg=cfg, support_mode=support_mode,
                    target_lon=target_lon, target_lat=target_lat,
                )
                metrics = _extreme_metrics(curve)
                metrics["date"] = date
                metrics["step"] = step
                bundles[key] = metrics
                del curve
                gc.collect()
            except Exception as exc:
                LOG.error("Failed %s %s: %s", event_name, key, exc)

        result[event_name] = bundles

    return result


def _compare(anchor_metrics: dict, new_metrics: dict) -> dict:
    """Compare new vs anchor per-bundle metrics. Returns structured comparison."""
    comparisons = {}

    for event_name in anchor_metrics:
        anchor_bundles = anchor_metrics[event_name]
        new_bundles = new_metrics.get(event_name, {})
        event_comp = []

        for key in sorted(anchor_bundles.keys()):
            anchor = anchor_bundles[key]
            new = new_bundles.get(key)
            if new is None:
                event_comp.append({"bundle": key, "status": "missing"})
                continue

            comp = {
                "bundle": key,
                "status": "ok",
            }
            for metric in ["mslp_980_990_fraction", "wind_gt_25_fraction", "mslp_min", "wind_max"]:
                a_val = anchor.get(metric, 0)
                n_val = new.get(metric, 0)
                comp[f"anchor_{metric}"] = a_val
                comp[f"new_{metric}"] = n_val
                if a_val != 0:
                    comp[f"ratio_{metric}"] = n_val / a_val
                else:
                    comp[f"ratio_{metric}"] = float("inf") if n_val > 0 else 1.0

            event_comp.append(comp)

        # Aggregate sums
        a_mslp_total = sum(b.get("mslp_980_990_count", 0) for b in anchor_bundles.values())
        n_mslp_total = sum(b.get("mslp_980_990_count", 0) for b in new_bundles.values() if b)
        a_wind_total = sum(b.get("wind_gt_25_count", 0) for b in anchor_bundles.values())
        n_wind_total = sum(b.get("wind_gt_25_count", 0) for b in new_bundles.values() if b)

        comparisons[event_name] = {
            "per_bundle": event_comp,
            "anchor_mslp_total": a_mslp_total,
            "new_mslp_total": n_mslp_total,
            "mslp_ratio": n_mslp_total / a_mslp_total if a_mslp_total else float("inf"),
            "anchor_wind_total": a_wind_total,
            "new_wind_total": n_wind_total,
            "wind_ratio": n_wind_total / a_wind_total if a_wind_total else float("inf"),
        }

    return comparisons


def _print_summary(comparisons: dict) -> None:
    """Print human-readable comparison summary."""
    for event_name, comp in comparisons.items():
        print(f"\n{'='*60}")
        print(f"  {event_name.upper()} — Proxy TC Extreme Comparison")
        print(f"{'='*60}")

        # Aggregate
        mslp_r = comp["mslp_ratio"]
        wind_r = comp["wind_ratio"]
        mslp_arrow = "+" if mslp_r > 1 else "-" if mslp_r < 1 else "="
        wind_arrow = "+" if wind_r > 1 else "-" if wind_r < 1 else "="

        print(f"\n  MSLP 980-990 hPa count: anchor={comp['anchor_mslp_total']}, "
              f"new={comp['new_mslp_total']} ({mslp_arrow}{abs(mslp_r - 1)*100:.1f}%)")
        print(f"  Wind >25 m/s count:     anchor={comp['anchor_wind_total']}, "
              f"new={comp['new_wind_total']} ({wind_arrow}{abs(wind_r - 1)*100:.1f}%)")

        # Per-bundle detail
        print(f"\n  {'Bundle':<20} {'MSLP frac ratio':>16} {'Wind frac ratio':>16} {'Min MSLP':>10} {'Max wind':>10}")
        print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*10} {'-'*10}")

        for b in comp["per_bundle"]:
            if b["status"] == "missing":
                print(f"  {b['bundle']:<20} {'MISSING':>16}")
                continue

            mslp_r = b["ratio_mslp_980_990_fraction"]
            wind_r = b["ratio_wind_gt_25_fraction"]
            mslp_str = f"{mslp_r:.2f}x" if mslp_r != float("inf") else "new>0"
            wind_str = f"{wind_r:.2f}x" if wind_r != float("inf") else "new>0"
            min_mslp = f"{b['new_mslp_min']:.1f}" if "new_mslp_min" in b else "?"
            max_wind = f"{b['new_wind_max']:.1f}" if "new_wind_max" in b else "?"
            print(f"  {b['bundle']:<20} {mslp_str:>16} {wind_str:>16} {min_mslp:>10} {max_wind:>10}")

    # Overall verdict
    print(f"\n{'='*60}")
    all_mslp = [c["mslp_ratio"] for c in comparisons.values()]
    all_wind = [c["wind_ratio"] for c in comparisons.values()]
    avg_mslp = sum(all_mslp) / len(all_mslp) if all_mslp else 1.0
    avg_wind = sum(all_wind) / len(all_wind) if all_wind else 1.0

    verdict = "COMPARABLE"
    if avg_mslp > 1.1 and avg_wind > 1.1:
        verdict = "STRONGER TC extremes than anchor"
    elif avg_mslp < 0.9 and avg_wind < 0.9:
        verdict = "WEAKER TC extremes than anchor"
    elif avg_mslp > 1.2 or avg_wind > 1.2:
        verdict = "POSSIBLY STRONGER (check per-event)"
    elif avg_mslp < 0.8 or avg_wind < 0.8:
        verdict = "POSSIBLY WEAKER (check per-event)"

    print(f"  VERDICT: {verdict}")
    print(f"  Avg MSLP ratio: {avg_mslp:.2f}x | Avg Wind ratio: {avg_wind:.2f}x")
    print(f"{'='*60}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser(description="Compare proxy TC extremes against anchor.")
    ap.add_argument("--predictions-dir", required=True, help="New run's predictions directory")
    ap.add_argument(
        "--anchor-ref",
        default="/home/ecm5702/scratch/per_bundle_tc_extremes_56b6c4e2_anchor.json",
        help="Path to anchor per-bundle diagnostic JSON",
    )
    ap.add_argument("--support-mode", default="native", choices=["native", "regridded"])
    ap.add_argument("--out", default="", help="Output comparison JSON path")
    args = ap.parse_args()

    anchor_path = Path(args.anchor_ref)
    if not anchor_path.exists():
        raise SystemExit(f"Anchor reference not found: {anchor_path}")

    predictions_dir = Path(args.predictions_dir)
    if not predictions_dir.exists():
        raise SystemExit(f"Predictions directory not found: {predictions_dir}")

    LOG.info("Loading anchor reference from %s", anchor_path)
    anchor_metrics = _load_anchor_proxy_metrics(anchor_path)

    LOG.info("Computing TC extremes for new predictions in %s", predictions_dir)
    new_metrics = _compute_metrics_from_predictions(predictions_dir, args.support_mode)

    LOG.info("Comparing...")
    comparisons = _compare(anchor_metrics, new_metrics)

    _print_summary(comparisons)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(comparisons, f, indent=2, default=str)
        LOG.info("Wrote comparison to %s", out_path)


if __name__ == "__main__":
    main()
