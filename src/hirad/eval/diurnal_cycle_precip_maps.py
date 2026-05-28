"""
Plots mean precipitation and wet-hour fraction maps for each hour of the diurnal cycle.

For each hour (00-23 UTC), all timesteps with that hour are averaged into
a single spatial map, producing 24 maps per source per variable:
  - mean precipitation (mm/h)
  - wet-hour fraction (% of timesteps where precip > wet_threshold)
"""
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from hirad.eval.eval_utils import (
    get_channel_indices,
    grid_cfg_from_cfg,
    load_generation_setup,
    parse_eval_cli,
    resolve_ts_dir,
)
from hirad.eval.plotting import plot_map, plot_map_precipitation


def main(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting diurnal cycle precipitation map generation")

    grid_cfg = grid_cfg_from_cfg(cfg)

    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps") / "diurnal_cycle_precip_maps"
    output_path.mkdir(parents=True, exist_ok=True)

    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    # conv_factor_hourly converts ERA5 accumulated precip (m) to mm/h
    conv_factor = cfg.get("conv_factor_hourly", 1000)
    wet_threshold = cfg.get("wet_threshold", 0.1)  # mm/h
    log_interval = cfg.get("log_interval", 24)

    logger.info(f"TP channel indices — output: {tp_out}, input: {tp_in}")
    logger.info(f"Wet-hour threshold: {wet_threshold} mm/h")
    logger.info(f"Processing {len(times)} timesteps")

    # Group timestep strings by UTC hour (no data loaded yet)
    times_by_hour: dict[int, list[str]] = defaultdict(list)
    for ts in times:
        times_by_hour[datetime.strptime(ts, "%Y%m%d-%H%M").hour].append(ts)

    hours_present = sorted(times_by_hour)
    logger.info(f"Hours present: {hours_present}")

    # Detect whether regression-prediction files exist (check first timestep)
    first_ts = times[0]
    first_dir = resolve_ts_dir(out_root, first_ts) / first_ts
    has_regression = (first_dir / f"{first_ts}-regression-prediction").exists()

    sources = [
        ('target',      'Target',               tp_out),
        ('baseline',    'Input',                tp_in),
        ('predictions', 'CorrDiff Ensemble Mean', tp_out),
    ]
    if has_regression:
        sources.append(('regression-prediction', 'Regression Prediction', tp_out))

    for source_key, source_label, tp_idx in sources:
        (output_path / source_key).mkdir(parents=True, exist_ok=True)
        (output_path / f"{source_key}_wethour").mkdir(parents=True, exist_ok=True)

    H, W = cfg.get("height"), cfg.get("width")

    for hour in hours_present:
        hour_times = times_by_hour[hour]
        logger.info(f"Hour {hour:02d}:00 UTC — loading {len(hour_times)} timesteps")

        # Accumulators for this hour only
        sums = {s[0]: np.zeros((H, W), dtype=np.float64) for s in sources}
        wet_sums = {s[0]: np.zeros((H, W), dtype=np.float64) for s in sources}
        count = 0

        for idx, ts in enumerate(hour_times, 1):
            ts_dir = resolve_ts_dir(out_root, ts) / ts

            target = np.asarray(torch.load(ts_dir / f"{ts}-target", weights_only=False)[tp_out]) * conv_factor
            sums['target'] += target
            wet_sums['target'] += (target > wet_threshold).astype(np.float64)

            baseline = np.asarray(torch.load(ts_dir / f"{ts}-baseline", weights_only=False)[tp_in]) * conv_factor
            sums['baseline'] += baseline
            wet_sums['baseline'] += (baseline > wet_threshold).astype(np.float64)

            preds = np.asarray(torch.load(ts_dir / f"{ts}-predictions", weights_only=False)[:, tp_out]) * conv_factor
            pred_mean = preds.mean(axis=0)
            sums['predictions'] += pred_mean
            # wet-hour frequency: fraction of members that are wet, then average over timesteps
            wet_sums['predictions'] += (preds > wet_threshold).mean(axis=0)

            if has_regression:
                reg = np.asarray(torch.load(ts_dir / f"{ts}-regression-prediction", weights_only=False)[tp_out]) * conv_factor
                sums['regression-prediction'] += reg
                wet_sums['regression-prediction'] += (reg > wet_threshold).astype(np.float64)

            count += 1
            if idx % log_interval == 0 or idx == len(hour_times):
                logger.info(f"  Loaded {idx}/{len(hour_times)} ({ts})")

        # Plot and immediately discard the accumulators
        for source_key, source_label, _ in sources:
            mean_map = sums[source_key] / count
            title = f"{source_label} — Mean Diurnal Precip {hour:02d}:00 UTC (n={count})"
            out_file = str(output_path / source_key / f"diurnal_mean_precip_{source_key}_{hour:02d}h")
            plot_map_precipitation(
                mean_map, out_file,
                title=title,
                threshold=0.01,
                rfac=1.0,
                grid_cfg=grid_cfg,
            )

            wet_map = wet_sums[source_key] / count * 100.0  # percent
            wh_title = f"{source_label} — Wet-Hour Fraction {hour:02d}:00 UTC (n={count})"
            wh_out_file = str(output_path / f"{source_key}_wethour" / f"diurnal_wethour_{source_key}_{hour:02d}h")
            plot_map(
                wet_map, wh_out_file,
                title=wh_title,
                label="Wet-Hour Fraction [%]",
                vmin=0, vmax=30,
                cmap="PuBu",
                extend="max",
                grid_cfg=grid_cfg,
            )

        del sums, wet_sums
        logger.info(f"Hour {hour:02d}:00 UTC — maps saved")

    logger.info("Diurnal cycle precipitation maps complete.")


if __name__ == '__main__':
    main(parse_eval_cli())
