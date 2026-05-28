"""
Plots the diurnal cycle of the all-hour 99th percentile of
precipitation, a somewhat reliable measure of the precipitation intensity.

Each hour, member and type is treaded separately, to conserve memory... but if the 
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

    logger.info("Starting computation for diurnal cycle of 99th-percentile of precipitation")
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

    # Storage for diurnal cycles
    pct99_mean = {}
    pct99_std = {}
    
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
        land_bool = land_mask.notnull().stack(space=('lat', 'lon'))
        da_land = da.stack(space=('lat', 'lon')).isel(space=land_bool.values)

        # Group by hour and compute 99th percentile over time, then spatial mean
        hourly_p99 = da_land.groupby('time.hour').quantile(0.99, dim='time')
        pct99_mean[mode] = hourly_p99.mean(dim='space')
            
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
    land_bool = land_mask.notnull().stack(space=('lat', 'lon'))
    pred_da_land = pred_da.stack(space=('lat', 'lon')).isel(space=land_bool.values)

    logger.info('Calculating 99th percentile for predictions')
    # Group by hour, compute 99th percentile across time, then spatial mean over land
    hourly_p99_by_member = pred_da_land.groupby('time.hour').quantile(0.99, dim='time').mean(dim='space')
    
    # Store ensemble statistics as xarray DataArrays
    pct99_mean['prediction'] = hourly_p99_by_member.mean(dim='member')
    pct99_std['prediction'] = hourly_p99_by_member.std(dim='member')
    
    # Prepare cyclic lists for plotting
    def cycle_fn(x):
        return x.values.tolist() + [x.values.tolist()[0]]
    
    logger.info("Preparing data for plotting")
    hrs_c = list(range(24)) + [0 + 24]
    pct99_lines = [
        cycle_fn(pct99_mean['target']),
        cycle_fn(pct99_mean['baseline']),
        (
            cycle_fn(pct99_mean['prediction']),
            cycle_fn(pct99_std['prediction'])
        )
    ]
    if 'regression-prediction' in pct99_mean:
        pct99_lines.append(cycle_fn(pct99_mean['regression-prediction']))

    # Plot combined diurnal 99th-percentile cycle
    labels = ['Target', 'Input', 'CorrDiff 99th Pct ± Std', 'Regression Prediction'] if 'regression-prediction' in pct99_mean else ['Target', 'Input', 'CorrDiff 99th Pct ± Std']
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    fn = output_path / 'diurnal_cycle_precip_99th_percentile.png'
    save_plot(
        hrs_c,
        pct99_lines,
        labels,
        'Precipitation (mm/day)',
        'Diurnal Cycle of 99th-Percentile Precipitation',
        fn
    )
    logger.info(f"Combined plot saved: {fn}")

if __name__ == '__main__':
    main(parse_eval_cli())