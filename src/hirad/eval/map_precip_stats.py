"""
Generates:
    - outputs/maps_mean/
    - outputs/maps_p99/
    - outputs/maps_p99.9/
    - outputs/maps_p99.99/
    - outputs/maps_Rx1hr/
    - outputs/maps_Rx1day/
    - outputs/maps_Rx5day/
    - outputs/maps_cdd/
    - outputs/maps_cwd/
    - outputs/maps_weth_freq/
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
from hirad.eval.plotting import plot_map_precipitation, plot_map, get_channel_indices, CONV_FACTOR, LOG_INTERVAL, WET_THRESHOLD


def process_data_for_stat(times, out_root, tp_channel, mode, stat_type, stat_param, logger):
    """Process data for a given mode and compute the specified statistic."""
    def consecutive_spell(condition):
        """Calculate longest consecutive spell where condition is True."""
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
        if stat_type == 'mean':
            return data.mean(dim='time')
        elif stat_type == 'quantile':
            return data.quantile(stat_param, dim='time')
        elif stat_type == 'Rx1hr':
            return data.max(dim='time')
        elif stat_type == 'Rx1day':
            # Maximum daily precipitation total
            daily = data.resample(time="1D").sum("time")
            return daily.max(dim='time')
        elif stat_type == 'Rx5day':
            # Maximum 5-consecutive-day precipitation total
            daily = data.resample(time="1D").sum("time")
            return daily.rolling(time=5, center=False).sum().max(dim='time')
        elif stat_type == 'cdd':  # Consecutive Dry Days (< 1 mm)
            daily = data.resample(time="1D").sum("time")
            return consecutive_spell(daily < 1.0)
        elif stat_type == 'cwd':  # Consecutive Wet Days (≥ 1 mm)
            daily = data.resample(time="1D").sum("time")
            return consecutive_spell(daily >= 1.0)
        elif stat_type == 'weth_freq':
            return (data / 24 > WET_THRESHOLD).mean(dim='time') * 100
        else:
            raise ValueError(f"Unsupported statistic type: {stat_type}")
    
    # Load data
    data_list = []
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing {mode} timestep {i+1}/{len(times)}: {ts}")
        
        data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False) * CONV_FACTOR
        if mode == 'predictions':
            tp_da = xr.DataArray(data[:, tp_channel], dims=['member', 'lat', 'lon'])
            data_list.append(tp_da)
        else:
            data_list.append(data[tp_channel])
    
    # Process data based on mode
    if mode == 'predictions':
        pred_da = xr.concat(data_list, dim='time').assign_coords({
            'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
        }).transpose('member', 'time', 'lat', 'lon')
        
        result_by_member = apply_statistic(pred_da, stat_type, stat_param)
        return result_by_member.mean(dim='member'), result_by_member, pred_da.shape[0]
    
    else:
        da = xr.DataArray(
            np.stack(data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        if mode == 'baseline':
            da = da / 6.0
        
        return apply_statistic(da, stat_type, stat_param), None, None


# Statistics configuration
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

def get_all_stat_configs():
    """Get all statistic configurations to generate."""
    return [
        {
            'stat_name': name,
            'title_stat': config['title'],
            'param': config.get('param'),  # Use get() to handle missing params
            **config
        }
        for name, config in STATISTICS_CONFIG.items()
    ]


def plot_stat_map(data, filename, stat_config, label):
    """Plot a single statistic map with appropriate styling."""
    if stat_config['type'] == 'weth_freq':
        plot_map(data, filename, title=f'{label}: {stat_config["title_stat"]} (%)',
                 label='Wet-Hour Frequency [%]', vmin=0, vmax=100, cmap='Blues')
    elif stat_config['type'] in ['cdd', 'cwd']:
        plot_map(data, filename, title=f'{label}: {stat_config["title_stat"]}',
                 label='Days', vmin=0, vmax=None, cmap='viridis')
    else:
        plot_map_precipitation(data, filename, title=f'{label}: {stat_config["title_stat"]} Precipitation',
                              threshold=stat_config['threshold'], rfac=1.0)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting precipitation statistics generation")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Processing {len(times)} timesteps")

    # Initialize dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )

    # Output root and channel indices
    out_root = Path(cfg.generation.io.output_path or './outputs')
    indices = get_channel_indices(dataset)
    tp_out, tp_in = indices['output']['tp'], indices['input'].get('tp', indices['output']['tp'])

    # Mode configuration
    modes = {
        'target': (tp_out, 'COSMO-2 Analysis'),
        'baseline': (tp_in, 'ERA5'), 
        'predictions': (tp_out, 'CorrDiff Ensemble Mean')
    }
    
    all_stat_configs = get_all_stat_configs()
    logger.info(f"Generating {len(all_stat_configs)} statistics for {len(modes)} modes")
    
    # Process each statistic
    for stat_config in all_stat_configs:
        logger.info(f"Processing {stat_config['title_stat']}...")
        
        # Process all modes for this statistic
        results = {}
        for mode, (tp_channel, label) in modes.items():
            result_mean, result_by_member, n_members = process_data_for_stat(
                times, out_root, tp_channel, mode, stat_config['type'], stat_config['param'], logger
            )
            results[mode] = (result_mean, result_by_member, n_members, label)

        # Create maps
        map_output_dir = out_root / f"maps_{stat_config['stat_name']}"
        map_output_dir.mkdir(parents=True, exist_ok=True)
        
        for mode, (result_mean, result_by_member, n_members, label) in results.items():
            # Main ensemble mean map
            plot_stat_map(result_mean.values, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label)
            
            # Individual ensemble member maps for predictions
            if mode == 'predictions' and result_by_member is not None:
                for member_idx in range(n_members):
                    member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
                    member_label = f'CorrDiff Member {member_idx+1}'
                    plot_stat_map(result_by_member[member_idx].values, member_filename, stat_config, member_label)
    
    logger.info("All precipitation statistics maps generated successfully")


if __name__ == '__main__':
    main()
