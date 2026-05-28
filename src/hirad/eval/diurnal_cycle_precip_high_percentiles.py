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

from hirad.eval.eval_utils import get_channel_indices, load_generation_setup, load_land_sea_mask, parse_eval_cli, resolve_ts_dir


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

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))
    land_bool = land_mask.notnull().stack(space=('lat', 'lon'))

    percentile_configs = [
        (0.99,   'p99',   '99th'),
        (0.999,  'p999',  '99.9th'),
        (0.9999, 'p9999', '99.99th'),
    ]

    # Storage for diurnal cycles: pct_mean[pct_key][mode], pct_std[pct_key]['prediction']
    pct_mean = {key: {} for _, key, _ in percentile_configs}
    pct_std  = {key: {} for _, key, _ in percentile_configs}

    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        data_list = []
        try:
            for ts in times:
                data = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-{mode}", weights_only=False)[tp_out if mode in ['target','regression-prediction'] else tp_in] * cfg.get("conv_factor")
                data_list.append(data)
        except:
            logger.error(f"Error loading data for mode {mode}. Skipping.")
            continue

        da = xr.DataArray(
            np.stack(data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times],
                    'lat': land_mask.coords['lat'], 'lon': land_mask.coords['lon']}
        )

        # Select only land pixels to avoid all-NaN slices in quantile
        da_land = da.stack(space=('lat', 'lon')).isel(space=land_bool.values)

        for q, key, _ in percentile_configs:
            hourly_pct = da_land.groupby('time.hour').quantile(q, dim='time')
            pct_mean[key][mode] = hourly_pct.mean(dim='space')

    # -- Predictions: compute per hour per member, then mean+std across members --
    logger.info("Processing predictions")
    
    # Load all prediction data at once into xarray
    pred_data_list = []
    for ts in times:
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False) * cfg.get("conv_factor")  # [n_members, n_channels, lat, lon]
        tp_data = preds[:, tp_out]  # [n_members, lat, lon]
        tp_da = xr.DataArray(tp_data, dims=['member', 'lat', 'lon'],
                             coords={'lat': land_mask.coords['lat'], 'lon': land_mask.coords['lon']})
        pred_data_list.append(tp_da)

    pred_da = xr.concat(pred_data_list, dim='time')  # [member, time, lat, lon]
    pred_da = pred_da.assign_coords({
        'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    })
    pred_da = pred_da.transpose('member', 'time', 'lat', 'lon')

    # Select only land pixels to avoid all-NaN slices in quantile
    pred_da_land = pred_da.stack(space=('lat', 'lon')).isel(space=land_bool.values)

    for q, key, label in percentile_configs:
        logger.info(f'Calculating {label} percentile for predictions')
        hourly_pct_by_member = pred_da_land.groupby('time.hour').quantile(q, dim='time').mean(dim='space')
        pct_mean[key]['prediction'] = hourly_pct_by_member.mean(dim='member')
        pct_std[key]['prediction']  = hourly_pct_by_member.std(dim='member')

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
            plot_labels.append(f'CorrDiff {label} Pct ± Std')
        if 'regression-prediction' in m:
            lines.append(cycle_fn(m['regression-prediction']))
            plot_labels.append('Regression Prediction')

        fn = output_path / f'diurnal_cycle_precip_{key}_percentile.png'
        save_plot(
            hrs_c,
            lines,
            plot_labels,
            'Precipitation (mm/day)',
            f'Diurnal Cycle of {label}-Percentile Precipitation',
            fn
        )
        logger.info(f"Plot saved: {fn}")

if __name__ == '__main__':
    main(parse_eval_cli())