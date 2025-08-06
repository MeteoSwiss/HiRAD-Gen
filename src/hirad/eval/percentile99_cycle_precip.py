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
from omegaconf import DictConfig, OmegaConf
import xarray as xr

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, CONV_FACTOR


def hour_of(dt: str, fmt: str = "%Y%m%d-%H%M") -> int:
    return datetime.strptime(dt, fmt).hour


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


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for diurnal cycle of 99th-percentile of precipitation")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Initialize dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    logger.info("Dataset and sampler initialized")

    # Output root and loader
    out_root = Path(cfg.generation.io.output_path or './outputs')
    def load(ts, fn):
        return torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask()

    # Storage for diurnal cycles
    pct99_mean = {}
    pct99_std = {}
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline']:
        logger.info(f"Processing mode: {mode}")
        
        data_list = []
        for ts in times:
            data = load(ts, f"{ts}-{mode}")[tp_out if mode == 'target' else tp_in] * land_mask
            data_list.append(data)
        
        da = xr.DataArray(
            np.stack(data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        
        # Group by hour and compute 99th percentile
        hourly_p99 = da.groupby('time.hour').quantile(0.99, dim='time')
        
        # Apply scaling factor for baseline
        if mode == 'baseline':
            hourly_p99 = hourly_p99 / 6.0
        
        pct99_mean[mode] = hourly_p99.mean(dim=['lat', 'lon'])
            
    # -- Predictions: compute per hour per member, then mean+std across members --
    logger.info("Processing predictions")
    
    # Load all prediction data at once into xarray
    pred_data_list = []
    for ts in times:
        preds = load(ts, f"{ts}-predictions")  # [n_members, n_channels, lat, lon]
        # Extract precipitation channel and convert to xarray for proper broadcasting
        tp_data = preds[:, tp_out]  # [n_members, lat, lon]
        tp_da = xr.DataArray(tp_data, dims=['member', 'lat', 'lon'])
        pred_data_list.append(tp_da * land_mask)  # apply mask
    
    pred_da = xr.concat(pred_data_list, dim='time')  # [n_members, time, lat, lon]
    pred_da = pred_da.assign_coords({
        'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    })
    # Transpose to get the expected dimension order: [member, time, lat, lon]
    pred_da = pred_da.transpose('member', 'time', 'lat', 'lon')
    
    # Group by hour, compute 99th percentile across time, then spatial mean
    hourly_p99_by_member = pred_da.groupby('time.hour').quantile(0.99, dim='time').mean(dim=['lat', 'lon'])
    
    # Store ensemble statistics as xarray DataArrays
    pct99_mean['prediction'] = hourly_p99_by_member.mean(dim='member')
    pct99_std['prediction'] = hourly_p99_by_member.std(dim='member')
    
    # Prepare cyclic lists for plotting
    def cycle_fn(x):
        return x.values.tolist() + [x.values.tolist()[0]]
    
    hrs_c = list(range(24)) + [0 + 24]
    pct99_lines = [
        cycle_fn(pct99_mean['target']),
        cycle_fn(pct99_mean['baseline']),
        (
            cycle_fn(pct99_mean['prediction']),
            cycle_fn(pct99_std['prediction'])
        )
    ]

    # Plot combined diurnal 99th-percentile cycle
    fn = out_root/'diurnal_cycle_precip_99th_percentile.png'
    save_plot(
        hrs_c,
        pct99_lines,
        ['COSMO-2  Analysis','ERA5','CorrDiff 99th Pct ± Std'],
        'Precipitation (mm/day)',
        'Diurnal Cycle of 99th-Percentile Precipitation',
        fn
    )
    logger.info(f"Combined plot saved: {fn}")

if __name__ == '__main__':
    main()
