import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import (
    plot_map_precipitation, plot_map, get_channel_indices, GridConfig
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
        return (data / 24 > cfg.get("wet_threshold")).mean(dim='time') * 100
    raise ValueError(f"Unsupported statistic type: {stat_type}")


def plot_stat_map(data, filename, stat_config, label, grid_cfg):
    """Plot a single statistic map with appropriate styling."""
    if stat_config['type'] == 'weth_freq':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} (%)',
            label='Wet-Hour Frequency [%]', vmin=0, vmax=30, cmap='PuBu', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'cdd':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Days', vmin=0, vmax=60, cmap='viridis', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'cwd':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Days', vmin=0, vmax=20, cmap='viridis', extend='max', grid_cfg=grid_cfg
        )
    else:
        plot_map_precipitation(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} Precipitation',
            threshold=stat_config['threshold'], rfac=1.0, grid_cfg=grid_cfg
        )


def main(cfg: dict):
    # Setup and config
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    grid_cfg = GridConfig(
        lat = np.arange(cfg.get("lat_start"), cfg.get("lat_end") + cfg.get("lat_step"), cfg.get("lat_step")),
        lon = np.arange(cfg.get("lon_start"), cfg.get("lon_end") + cfg.get("lon_step"), cfg.get("lon_step")),
        height = cfg.get("height"),
        width = cfg.get("width"),
        relax_zone = cfg.get("relax_zone")
    )

    generation_dir = cfg.get("inference_output_dir", None)
    if generation_dir is None:
        logger.error("No inference_output_dir specified in config.")
        return
    
    if not Path(generation_dir).exists() or not Path(generation_dir).is_dir():
        logger.error(f"Inference output directory {generation_dir} does not exist or is not a directory.")
        return

    generation_config_path = Path(generation_dir) / ".hydra" / "config.yaml"
    if not generation_config_path.exists():
        logger.error(f"Generation config file {generation_config_path} does not exist.")
        return

    with open(generation_config_path, "r") as f:
        gen_cfg = yaml.safe_load(f)

    logger.info("Starting precipitation statistics generation")
    if cfg.get("times_range", None):
        times = get_time_from_range(cfg.get("times_range"), time_format="%Y%m%d-%H%M")
    elif cfg.get("times", None):
        times = cfg.get("times")
    elif gen_cfg.get("generation").get("times_range", None):
        times = get_time_from_range(gen_cfg.get("generation").get("times_range"), time_format="%Y%m%d-%H%M")
    elif gen_cfg.get("generation").get("times", None):
        times = gen_cfg.get("generation").get("times")
    else:
        logger.error("No times or times_range specified in config or generation config.")
        return
    logger.info(f"Processing {len(times)} timesteps")

    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.pop("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
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
        'target': (tp_out, 'Target'),
        'baseline': (tp_in, 'Input'),
        'regression-prediction': (tp_out, 'Regression Prediction') 
    }
    logger.info(f"Generating {len(stat_configs)} statistics for {len(basic_modes)} basic modes + predictions")

    for mode, (tp_channel, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        # Load all timesteps for this mode
        data_list = []
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False) * cfg.get("conv_factor")
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
            map_output_dir = output_path / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_stat_map(result.values, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label, grid_cfg)

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
            if i % cfg.get("log_interval") == 0:
                logger.info(f"Loading prediction member {member_idx} timestep {i+1}/{len(times)}: {ts}")
            pred_data = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False) * cfg.get("conv_factor")
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
            map_output_dir = output_path / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'CorrDiff Member {member_idx+1}'
            plot_stat_map(member_result.values, member_filename, stat_config, member_label, grid_cfg)

    logger.info("All precipitation statistics maps generated successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)