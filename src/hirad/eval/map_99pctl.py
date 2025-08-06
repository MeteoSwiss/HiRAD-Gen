"""
Plots maps of the 99th percentile of precipitation over the entire time period.

This script computes the all-time 99th percentile of precipitation for each grid point
and creates maps for target (COSMO-2), baseline (ERA5), and predictions (CorrDiff).
"""
import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import xarray as xr

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import plot_map_precipitation, get_channel_indices

# Constants
CONV_FACTOR = 100 * 24   # Convert meters to mm/day
LOG_INTERVAL = 24    # Log progress every N timesteps


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for 99th percentile precipitation maps")
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

    # -- Process target --
    logger.info("Processing target (COSMO-2)")
    target_data_list = []
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing target timestep {i+1}/{len(times)}: {ts}")
        data = load(ts, f"{ts}-target")[tp_out]
        target_data_list.append(data)
    
    target_da = xr.DataArray(
        np.stack(target_data_list, axis=0),
        dims=['time', 'lat', 'lon'],
        coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
    )
    
    # Compute 99th percentile across all time steps
    target_p99 = target_da.quantile(0.99, dim='time')
    logger.info("Target 99th percentile computed")

    # -- Process baseline --
    logger.info("Processing baseline (ERA5)")
    baseline_data_list = []
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing baseline timestep {i+1}/{len(times)}: {ts}")
        data = load(ts, f"{ts}-baseline")[tp_in]
        baseline_data_list.append(data)
    
    baseline_da = xr.DataArray(
        np.stack(baseline_data_list, axis=0),
        dims=['time', 'lat', 'lon'],
        coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
    )
    
    # Apply scaling factor for baseline and compute 99th percentile
    baseline_p99 = (baseline_da / 6.0).quantile(0.99, dim='time')
    logger.info("Baseline 99th percentile computed")

    # -- Process predictions --
    logger.info("Processing predictions (CorrDiff)")
    pred_data_list = []
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing predictions timestep {i+1}/{len(times)}: {ts}")
        preds = load(ts, f"{ts}-predictions")  # [n_members, n_channels, lat, lon]
        # Extract precipitation channel
        tp_data = preds[:, tp_out]  # [n_members, lat, lon]
        tp_da = xr.DataArray(tp_data, dims=['member', 'lat', 'lon'])
        pred_data_list.append(tp_da)
    
    pred_da = xr.concat(pred_data_list, dim='time')  # [member, time, lat, lon]
    pred_da = pred_da.assign_coords({
        'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    })
    pred_da = pred_da.transpose('member', 'time', 'lat', 'lon')
    
    # Compute 99th percentile across time for each member, then ensemble mean
    pred_p99_by_member = pred_da.quantile(0.99, dim='time')
    pred_p99_mean = pred_p99_by_member.mean(dim='member')
    logger.info("Predictions 99th percentile computed")

    # Create output directory
    map_output_dir = out_root / 'maps_99th_percentile'
    map_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot maps using the precipitation-specific plotting function
    logger.info("Creating precipitation maps")
    
    # Target map
    plot_map_precipitation(
        target_p99.values,
        str(map_output_dir / 'target_99th_percentile'),
        title='COSMO-2 Analysis: 99th Percentile Precipitation',
        threshold=0.1,
        rfac=1.0  # Already converted to mm/day
    )
    logger.info("Target map saved")

    # Baseline map
    plot_map_precipitation(
        baseline_p99.values,
        str(map_output_dir / 'baseline_99th_percentile'),
        title='ERA5: 99th Percentile Precipitation',
        threshold=0.1,
        rfac=1.0  # Already converted to mm/day
    )
    logger.info("Baseline map saved")

    # Prediction ensemble mean map
    plot_map_precipitation(
        pred_p99_mean.values,
        str(map_output_dir / 'prediction_ensmean_99th_percentile'),
        title='CorrDiff Ensemble Mean: 99th Percentile Precipitation',
        threshold=0.1,
        rfac=1.0  # Already converted to mm/day
    )
    logger.info("Prediction ensemble mean map saved")

    # Individual ensemble member maps
    logger.info("Creating individual ensemble member maps")
    n_members = pred_p99_by_member.shape[0]
    for member_idx in range(n_members):
        plot_map_precipitation(
            pred_p99_by_member[member_idx].values,
            str(map_output_dir / f'prediction_member_{member_idx:02d}_99th_percentile'),
            title=f'CorrDiff Member {member_idx+1}: 99th Percentile Precipitation',
            threshold=0.1,
            rfac=1.0  # Already converted to mm/day
        )
    logger.info(f"Individual ensemble member maps saved ({n_members} members)")
    
    logger.info(f"All maps saved to {map_output_dir}")
    logger.info("99th percentile precipitation mapping completed successfully")


if __name__ == '__main__':
    main()