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
from hirad.eval.plotting import (
    plot_map_precipitation, plot_map, get_channel_indices,
    CONV_FACTOR, LOG_INTERVAL, WET_THRESHOLD
)


def consecutive_spell(condition):
    """Return longest consecutive spell where condition is True (per gridpoint)."""
    def _spell_length(x):
        x = np.asarray(x, dtype=bool)
        if len(x) == 0:
            return 0
        runs = np.diff(np.concatenate(([False], x, [False])).astype(int))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        return int(np.max(ends - starts)) if len(starts) > 0 else 0
    return xr.apply_ufunc(_spell_length, condition, input_core_dims=[['time']], vectorize=True)


def apply_statistic(data, stat_type, stat_param):
    """Apply a statistic to the data along the time dimension."""
    if stat_type == 'mean':
        return data.mean(dim='time')
    if stat_type == 'quantile':
        return data.quantile(stat_param, dim='time')
    if stat_type == 'Rx1hr':
        return data.max(dim='time')
    if stat_type == 'Rx1day':
        daily = data.resample(time="1D").sum("time")
        return daily.max(dim='time')
    if stat_type == 'Rx5day':
        daily = data.resample(time="1D").sum("time")
        return daily.rolling(time=5, center=False).sum().max(dim='time')
    if stat_type == 'cdd':
        daily = data.resample(time="1D").sum("time")
        return consecutive_spell(daily < 1.0)
    if stat_type == 'cwd':
        daily = data.resample(time="1D").sum("time")
        return consecutive_spell(daily >= 1.0)
    if stat_type == 'weth_freq':
        return (data / 24 > WET_THRESHOLD).mean(dim='time') * 100
    raise ValueError(f"Unsupported statistic type: {stat_type}")


def plot_stat_map(data, filename, stat_config, label):
    """Plot a single statistic map with appropriate styling."""
    if stat_config['type'] == 'weth_freq':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} (%)',
            label='Wet-Hour Frequency [%]', vmin=0, vmax=10, cmap='PuBu', extend='max'
        )
    elif stat_config['type'] == 'cdd':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Days', vmin=0, vmax=60, cmap='viridis', extend='max'
        )
    elif stat_config['type'] == 'cwd':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Days', vmin=0, vmax=20, cmap='viridis', extend='max'
        )
    else:
        plot_map_precipitation(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} Precipitation',
            threshold=stat_config['threshold'], rfac=1.0
        )


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup and config
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting precipitation statistics generation")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Processing {len(times)} timesteps")

    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    out_root = Path(cfg.generation.io.output_path or './outputs')
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)

    # Statistic configuration
    STATISTICS_CONFIG = {
        'mean': {'type': 'mean', 'threshold': 0.01, 'title': 'Mean'},
        'p99': {'type': 'quantile', 'param': 0.99, 'threshold': 0.1, 'title': '99th Percentile'},
        'p99.9': {'type': 'quantile', 'param': 0.999, 'threshold': 0.1, 'title': '99.9th Percentile'},
        'p99.99': {'type': 'quantile', 'param': 0.9999, 'threshold': 0.1, 'title': '99.99th Percentile'},
        'Rx1hr': {'type': 'Rx1hr', 'threshold': 0.1, 'title': 'Maximum (Rx1hr)'},
        'Rx1day': {'type': 'Rx1day', 'threshold': 0.1, 'title': 'Maximum 1-day Amount (Rx1day)'},
        'Rx5day': {'type': 'Rx5day', 'threshold': 0.1, 'title': 'Maximum 5-day Total (Rx5day)'},
        'cdd': {'type': 'cdd', 'threshold': 0.1, 'title': 'Consecutive Dry Days (CDD)'},
        'cwd': {'type': 'cwd', 'threshold': 0.1, 'title': 'Consecutive Wet Days (CWD)'},
        'weth_freq': {'type': 'weth_freq', 'threshold': 0.01, 'title': 'Wet-Hour Frequency'}
    }
    stat_configs = [
        {
            'stat_name': name,
            'title_stat': config['title'],
            'param': config.get('param'),
            **config
        }
        for name, config in STATISTICS_CONFIG.items()
    ]

    # Target and baseline modes
    basic_modes = {
        'target': (tp_out, 'COSMO-2 Analysis'),
        'baseline': (tp_in, 'ERA5'),
        'regression-prediction': (tp_out, 'Regression Prediction') 
    }
    logger.info(f"Generating {len(stat_configs)} statistics for {len(basic_modes)} basic modes + predictions")

    for mode, (tp_channel, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        # Load all timesteps for this mode
        data_list = []
        try:
            for i, ts in enumerate(times):
                if i % LOG_INTERVAL == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False) * CONV_FACTOR
                data_list.append(data[tp_channel])
        except:
            logger.warning(f"{mode} not available, skipping")
            continue
        mode_data = xr.DataArray(
            np.stack(data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        # if mode == 'baseline':
        #     mode_data = mode_data / 6.0
        # Compute and plot all statistics for this mode
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            result = apply_statistic(mode_data, stat_config['type'], stat_config['param'])
            map_output_dir = out_root / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_stat_map(result.values, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label)

    # Predictions mode: process each member separately to save memory
    logger.info("Processing predictions mode...")
    data = torch.load(out_root/times[0]/f"{times[0]}-predictions", weights_only=False)
    n_members = data.shape[0]
    logger.info(f"Found {n_members} ensemble members")
    
    for member_idx in range(n_members):
        logger.info(f"Processing prediction member {member_idx+1}/{n_members}")
        # Load all timesteps for this member
        data_list = []
        for i, ts in enumerate(times):
            if i % LOG_INTERVAL == 0:
                logger.info(f"Loading prediction member {member_idx} timestep {i+1}/{len(times)}: {ts}")
            pred_data = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False) * CONV_FACTOR
            data_list.append(pred_data[member_idx, tp_out])
        member_data = xr.DataArray(
            np.stack(data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        
        # Compute and plot all statistics for this member
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
            member_result = apply_statistic(member_data, stat_config['type'], stat_config['param'])
            
            # Create map
            map_output_dir = out_root / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'CorrDiff Member {member_idx+1}'
            plot_stat_map(member_result.values, member_filename, stat_config, member_label)

    logger.info("All precipitation statistics maps generated successfully")


if __name__ == '__main__':
    main()
