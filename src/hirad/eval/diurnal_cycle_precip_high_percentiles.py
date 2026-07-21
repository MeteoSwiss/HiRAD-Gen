"""
Plots the diurnal cycle of the all-hour 99th, 99.9th, and 99.99th percentiles of
precipitation, a somewhat reliable measure of the precipitation intensity.

Each hour, member and type is treated separately, to conserve memory... but if the 
period is long, this can still be a lot of data and thus an OOM error can occur.
"""
import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.eval.eval_utils import get_channel_indices, load_generation_setup, load_land_sea_mask, relax_zone_interior_mask, parse_eval_cli, precip_conv_factor, resolve_ts_dir, FONT_SIZE

# Presentation-sized fonts for all figures in this script.
plt.rcParams.update(FONT_SIZE)


def save_plot(hours, lines, labels, ylabel, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    for data, label in zip(lines, labels):
        if isinstance(data, tuple):  # (mean, std)
            mean, std = data
            lower = np.maximum(np.array(mean) - std, 0)
            upper = np.array(mean) + std
            line, = plt.plot(hours, mean, label=label)
            plt.fill_between(hours, lower, upper, alpha=0.3, color=line.get_color())
        else:
            plt.plot(hours, data, label=label)
    plt.xlabel('Hour (UTC)')
    plt.xticks(range(0,25,3))
    plt.xlim(0,24)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(cfg: dict):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for diurnal cycle of high percentiles of precipitation")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Output root
    out_root = Path(generation_dir)

    # Find channel indices
    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask (sea = NaN); the relaxation zone is dropped separately.
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))
    land_mask = land_mask.where(relax_zone_interior_mask(cfg.get("height"), cfg.get("width"), cfg.get("relax_zone")))
    land_bool = land_mask.notnull().stack(space=('lat', 'lon'))

    percentile_configs = [
        (0.99,   'p99',   '99th'),
        (0.999,  'p999',  '99.9th'),
        (0.9999, 'p9999', '99.99th'),
    ]

    # Storage for diurnal cycles: pct_mean[pct_key][mode], pct_std[pct_key]['prediction']
    pct_mean = {key: {} for _, key, _ in percentile_configs}
    pct_std  = {key: {} for _, key, _ in percentile_configs}
    conv_factor = precip_conv_factor(cfg)  # mm/h
    land_idx = land_bool.values  # 1D boolean mask over flattened (lat, lon)
    n_land = int(land_idx.sum())
    quantiles = np.array([q for q, _, _ in percentile_configs])
    logger.info(f"Land pixels: {n_land} / {land_idx.size} ({100 * n_land / land_idx.size:.1f}%)")

    # Group timesteps by hour-of-day so we never hold all timesteps in memory at once.
    times_by_hour = {}
    for ts in times:
        hour = datetime.strptime(ts, "%Y%m%d-%H%M").hour
        times_by_hour.setdefault(hour, []).append(ts)
    sorted_hours = sorted(times_by_hour)
    counts_per_hour = {h: len(times_by_hour[h]) for h in sorted_hours}
    logger.info(f"Grouped {len(times)} timesteps into {len(sorted_hours)} hours; timesteps/hour: {counts_per_hour}")

    def land_values(arr):
        """Flatten a (lat, lon) array and keep only land pixels as float32."""
        return np.asarray(arr, dtype=np.float32).reshape(-1)[land_idx]

    # -- Process target, baseline and regression-prediction --
    # For each hour we collect only land pixels across that hour's timesteps, then take
    # the spatial-mean of the per-hour quantiles. Memory scales with one hour's data.
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        ch = tp_out if mode in ['target', 'regression-prediction'] else tp_in

        per_hour_pct = {key: [] for _, key, _ in percentile_configs}
        failed = False
        for hi, hour in enumerate(sorted_hours):
            n_ts = counts_per_hour[hour]
            logger.info(
                f"[{mode}] hour {hour:02d} ({hi + 1}/{len(sorted_hours)}): "
                f"loading {n_ts} timesteps"
            )
            hour_vals = []
            try:
                for ts in times_by_hour[hour]:
                    data = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-{mode}", weights_only=False)[ch]
                    hour_vals.append(land_values(data) * conv_factor)
            except Exception as exc:
                logger.error(f"Error loading data for mode {mode} at hour {hour:02d}: {exc!r}. Skipping mode.")
                failed = True
                break

            stacked = np.stack(hour_vals, axis=0)  # [n_times_this_hour, n_land]
            del hour_vals
            logger.info(
                f"[{mode}] hour {hour:02d}: stacked array {stacked.shape}"
            )
            # quantile over time, then mean over space -> one value per quantile for this hour
            q_vals = np.nanquantile(stacked, quantiles, axis=0)  # [n_q, n_land]
            del stacked
            q_means = np.nanmean(q_vals, axis=1)  # [n_q]
            for (_, key, _), val in zip(percentile_configs, q_means):
                per_hour_pct[key].append(val)

        if failed:
            continue

        for _, key, _ in percentile_configs:
            pct_mean[key][mode] = xr.DataArray(
                np.array(per_hour_pct[key]), dims=['hour'], coords={'hour': sorted_hours}
            )
        logger.info(f"Finished mode: {mode}")

    # -- Predictions: compute per hour per member, then mean+std across members --
    logger.info("Processing predictions")

    pred_hour_mean = {key: [] for _, key, _ in percentile_configs}
    pred_hour_std = {key: [] for _, key, _ in percentile_configs}

    for hi, hour in enumerate(sorted_hours):
        n_ts = counts_per_hour[hour]
        logger.info(
            f"[predictions] hour {hour:02d} ({hi + 1}/{len(sorted_hours)}): "
            f"loading {n_ts} timesteps"
        )
        # Accumulate land pixels per member: member -> list of [n_land] arrays over this hour's times
        member_vals = None
        for ts in times_by_hour[hour]:
            preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False)  # [n_members, n_channels, lat, lon]
            tp_data = np.asarray(preds[:, tp_out], dtype=np.float32) * conv_factor  # [n_members, lat, lon]
            n_members = tp_data.shape[0]
            if member_vals is None:
                member_vals = [[] for _ in range(n_members)]
            flat = tp_data.reshape(n_members, -1)[:, land_idx]  # [n_members, n_land]
            for m in range(n_members):
                member_vals[m].append(flat[m])
            del preds, tp_data, flat

        n_members = len(member_vals)
        logger.info(
            f"[predictions] hour {hour:02d}: {n_members} members x {n_ts} timesteps"
        )

        # Per-member: quantile over this hour's timesteps, then spatial mean -> [n_q] per member
        per_member_q = []
        for vals in member_vals:
            stacked = np.stack(vals, axis=0)  # [n_times_this_hour, n_land]
            q_vals = np.nanquantile(stacked, quantiles, axis=0)  # [n_q, n_land]
            del stacked
            per_member_q.append(np.nanmean(q_vals, axis=1))  # [n_q]
        per_member_arr = np.stack(per_member_q, axis=0)  # [n_members, n_q]
        del member_vals

        q_mean_over_members = per_member_arr.mean(axis=0)  # [n_q]
        q_std_over_members = per_member_arr.std(axis=0)  # [n_q]
        for i, (_, key, _) in enumerate(percentile_configs):
            pred_hour_mean[key].append(q_mean_over_members[i])
            pred_hour_std[key].append(q_std_over_members[i])

    logger.info("Finished predictions")

    for _, key, _ in percentile_configs:
        pct_mean[key]['prediction'] = xr.DataArray(
            np.array(pred_hour_mean[key]), dims=['hour'], coords={'hour': sorted_hours}
        )
        pct_std[key]['prediction'] = xr.DataArray(
            np.array(pred_hour_std[key]), dims=['hour'], coords={'hour': sorted_hours}
        )

    # Prepare cyclic lists for plotting
    def cycle_fn(x):
        vals = x.values.tolist()
        return vals + [vals[0]]

    logger.info("Preparing data for plotting")
    hrs_c = list(range(24)) + [24]
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)

    for _, key, label in percentile_configs:
        m = pct_mean[key]
        s = pct_std[key]

        lines = []
        plot_labels = []

        if 'target' in m:
            lines.append(cycle_fn(m['target']))
            plot_labels.append('Target')
        if 'baseline' in m:
            lines.append(cycle_fn(m['baseline']))
            plot_labels.append('Input')
        if 'prediction' in m:
            lines.append((cycle_fn(m['prediction']), cycle_fn(s['prediction'])))
            plot_labels.append(f'Pred {label} Pct ± Std')
        if 'regression-prediction' in m:
            lines.append(cycle_fn(m['regression-prediction']))
            plot_labels.append('Regression Prediction')

        fn = output_path / f'diurnal_cycle_precip_{key}_percentile.png'
        save_plot(
            hrs_c,
            lines,
            plot_labels,
            'Precipitation (mm/h)',
            f'Diurnal Cycle of {label}-Percentile Precipitation',
            fn
        )
        logger.info(f"Plot saved: {fn}")

if __name__ == '__main__':
    main(parse_eval_cli())