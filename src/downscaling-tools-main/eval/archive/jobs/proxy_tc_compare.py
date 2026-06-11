#!/usr/bin/env python
"""
Proxy TC extreme comparison.

Compares TC extreme metrics from the canonical 10 proxy date/step files against
the anchor baseline, producing a concise pass/warn/fail signal for quick
screening.

Usage:
    python -m eval.jobs.proxy_tc_compare \
        --proxy-predictions-dir /home/ecm5702/scratch/eval/<proxy_run>/predictions \
        --anchor-json /home/ecm5702/scratch/per_bundle_tc_extremes_56b6c4e2_anchor.json \
        --out-json /home/ecm5702/scratch/proxy_compare_<run_id>.json

    # Or compare two proxy diagnostics directly:
    python -m eval.jobs.proxy_tc_compare \
        --proxy-json /home/ecm5702/scratch/proxy_extremes_run1.json \
        --anchor-json /home/ecm5702/scratch/per_bundle_tc_extremes_56b6c4e2_anchor.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

# Canonical proxy date/step pairs (top 10 by combined Idalia+Franklin extreme signal on anchor)
PROXY_BUNDLES = [
    "20230829_step024",
    "20230828_step048",
    "20230829_step048",
    "20230828_step024",
    "20230830_step024",
    "20230828_step072",
    "20230827_step072",
    "20230830_step048",
    "20230829_step072",
    "20230827_step048",
]

# Thresholds for pass/warn/fail (relative deviation from anchor)
WARN_THRESHOLD = 0.15   # >15% deviation = warn
FAIL_THRESHOLD = 0.30   # >30% deviation = fail


def _bundle_key(row: dict) -> str:
    return f"{row['date']}_step{row['step']:03d}"


def _extract_proxy_metrics(event_data: dict) -> dict[str, dict]:
    """Extract per-bundle metrics for proxy bundles only."""
    out = {}
    for row in event_data["per_bundle"]:
        key = _bundle_key(row)
        if key in PROXY_BUNDLES:
            out[key] = row
    return out


def compare_events(
    anchor_data: dict,
    proxy_data: dict,
) -> dict:
    """Compare proxy vs anchor TC extreme metrics per event."""
    results = {"events": {}, "overall_verdict": "PASS"}

    for event_name in anchor_data["events"]:
        if event_name not in proxy_data["events"]:
            results["events"][event_name] = {"status": "MISSING", "detail": "event not found in proxy"}
            results["overall_verdict"] = "FAIL"
            continue

        anchor_event = anchor_data["events"][event_name]
        proxy_event = proxy_data["events"][event_name]

        anchor_bundles = _extract_proxy_metrics(anchor_event)
        proxy_bundles = _extract_proxy_metrics(proxy_event)

        # Aggregate counts across proxy bundles for each metric
        metrics_to_compare = [
            ("mslp_980_990_count", "MSLP 980-990 count"),
            ("wind_gt_25_count", "Wind >25 count"),
        ]
        fraction_metrics = [
            ("mslp_980_990_fraction", "MSLP 980-990 fraction"),
            ("wind_gt_25_fraction", "Wind >25 fraction"),
        ]

        event_result = {
            "status": "PASS",
            "per_bundle": {},
            "aggregate": {},
        }

        # Per-bundle comparison
        max_deviation = 0.0
        for bundle_key in PROXY_BUNDLES:
            a = anchor_bundles.get(bundle_key)
            p = proxy_bundles.get(bundle_key)

            if a is None:
                event_result["per_bundle"][bundle_key] = {"status": "MISSING_ANCHOR"}
                continue
            if p is None:
                event_result["per_bundle"][bundle_key] = {"status": "MISSING_PROXY"}
                event_result["status"] = "WARN"
                continue

            bundle_result = {}
            for metric_key, label in fraction_metrics:
                anchor_val = a[metric_key]
                proxy_val = p[metric_key]

                if anchor_val == 0 and proxy_val == 0:
                    deviation = 0.0
                elif anchor_val == 0:
                    deviation = 1.0 if proxy_val > 0 else 0.0
                else:
                    deviation = abs(proxy_val - anchor_val) / anchor_val

                bundle_result[metric_key] = {
                    "anchor": anchor_val,
                    "proxy": proxy_val,
                    "relative_deviation": round(deviation, 4),
                }
                max_deviation = max(max_deviation, deviation)

            event_result["per_bundle"][bundle_key] = bundle_result

        # Aggregate comparison (sum across proxy bundles)
        for metric_key, label in metrics_to_compare:
            anchor_sum = sum(anchor_bundles[k][metric_key] for k in anchor_bundles)
            proxy_sum = sum(proxy_bundles[k][metric_key] for k in proxy_bundles)

            if anchor_sum == 0 and proxy_sum == 0:
                agg_dev = 0.0
            elif anchor_sum == 0:
                agg_dev = 1.0 if proxy_sum > 0 else 0.0
            else:
                agg_dev = abs(proxy_sum - anchor_sum) / anchor_sum

            event_result["aggregate"][metric_key] = {
                "anchor": anchor_sum,
                "proxy": proxy_sum,
                "relative_deviation": round(agg_dev, 4),
                "label": label,
            }
            max_deviation = max(max_deviation, agg_dev)

        # Tail percentile comparison
        for tail_metric in ["mslp_min", "mslp_p01", "wind_max", "wind_p99"]:
            anchor_vals = [anchor_bundles[k][tail_metric] for k in anchor_bundles if tail_metric in anchor_bundles[k]]
            proxy_vals = [proxy_bundles[k][tail_metric] for k in proxy_bundles if tail_metric in proxy_bundles[k]]

            if anchor_vals and proxy_vals:
                if tail_metric.startswith("mslp"):
                    # Lower is more extreme for MSLP
                    a_extreme = min(anchor_vals)
                    p_extreme = min(proxy_vals)
                else:
                    # Higher is more extreme for wind
                    a_extreme = max(anchor_vals)
                    p_extreme = max(proxy_vals)

                if a_extreme != 0:
                    dev = abs(p_extreme - a_extreme) / abs(a_extreme)
                else:
                    dev = 0.0

                event_result["aggregate"][tail_metric] = {
                    "anchor_extreme": round(a_extreme, 2),
                    "proxy_extreme": round(p_extreme, 2),
                    "relative_deviation": round(dev, 4),
                }

        event_result["max_deviation"] = round(max_deviation, 4)
        if max_deviation > FAIL_THRESHOLD:
            event_result["status"] = "FAIL"
        elif max_deviation > WARN_THRESHOLD:
            event_result["status"] = "WARN"

        results["events"][event_name] = event_result

        if event_result["status"] == "FAIL":
            results["overall_verdict"] = "FAIL"
        elif event_result["status"] == "WARN" and results["overall_verdict"] == "PASS":
            results["overall_verdict"] = "WARN"

    return results


def print_summary(results: dict) -> None:
    """Print human-readable comparison summary."""
    verdict = results["overall_verdict"]
    symbol = {"PASS": "+", "WARN": "~", "FAIL": "!"}[verdict]
    print(f"\n[{symbol}] PROXY TC COMPARISON: {verdict}")
    print("=" * 60)

    for event_name, event_data in results["events"].items():
        if isinstance(event_data, dict) and "status" in event_data:
            status = event_data["status"]
            sym = {"PASS": "+", "WARN": "~", "FAIL": "!", "MISSING": "!"}[status]
            print(f"\n  [{sym}] {event_name.upper()}: {status}")

            if "aggregate" in event_data:
                for key, agg in event_data["aggregate"].items():
                    if "label" in agg:
                        dev_pct = agg["relative_deviation"] * 100
                        print(f"      {agg['label']:25s}  anchor={agg['anchor']:5d}  proxy={agg['proxy']:5d}  dev={dev_pct:+.1f}%")
                    elif "anchor_extreme" in agg:
                        dev_pct = agg["relative_deviation"] * 100
                        print(f"      {key:25s}  anchor={agg['anchor_extreme']:10.2f}  proxy={agg['proxy_extreme']:10.2f}  dev={dev_pct:+.1f}%")

            if "max_deviation" in event_data:
                print(f"      max deviation: {event_data['max_deviation']*100:.1f}%")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare proxy TC extremes against anchor baseline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--proxy-predictions-dir",
        help="Path to proxy predictions directory (will run diagnostic first).",
    )
    group.add_argument(
        "--proxy-json",
        help="Path to pre-computed proxy diagnostic JSON.",
    )
    parser.add_argument("--anchor-json", required=True, help="Path to anchor diagnostic JSON.")
    parser.add_argument("--out-json", default="", help="Output comparison JSON.")
    parser.add_argument("--support-mode", choices=["native", "regridded"], default="native")
    parser.add_argument(
        "--analysis-expid",
        default="",
        help="Analysis truth expid passed to regridded diagnostics.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load anchor
    with open(args.anchor_json, encoding="utf-8") as f:
        anchor_data = json.load(f)

    # Load or compute proxy
    if args.proxy_json:
        with open(args.proxy_json, encoding="utf-8") as f:
            proxy_data = json.load(f)
    else:
        from eval.jobs.diagnose_per_bundle_tc_extremes import diagnose_per_bundle

        LOG.info("Running per-bundle diagnostic on proxy predictions...")
        proxy_data = diagnose_per_bundle(
            args.proxy_predictions_dir,
            support_mode=args.support_mode,
            analysis_expid=args.analysis_expid or None,
        )

    # Compare
    results = compare_events(anchor_data, proxy_data)

    # Write output
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        LOG.info("Wrote comparison to %s", out_path)

    print_summary(results)

    # Exit code reflects verdict
    if results["overall_verdict"] == "FAIL":
        sys.exit(1)
    elif results["overall_verdict"] == "WARN":
        sys.exit(0)  # warn but don't fail
    sys.exit(0)


if __name__ == "__main__":
    main()
