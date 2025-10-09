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
from hirad.eval.plotting import plot_map, get_channel_indices, LOG_INTERVAL


def compute_wind_speed(u, v):
    """Compute wind speed from U and V components."""
    return np.hypot(u, v)


def compute_wind_direction(u, v):
    """Compute wind direction in degrees (meteorological convention: direction FROM which wind blows).
    
    Returns angle in degrees: 0° = North, 90° = East, 180° = South, 270° = West
    """
    # atan2(u, v) gives mathematical angle, convert to meteorological
    direction = np.rad2deg(np.arctan2(u, v)) + 180
    direction = np.mod(direction, 360)
    return direction


def circular_mean_direction(directions, weights=None):
    """Calculate circular mean of wind directions in degrees.
    
    Args:
        directions: Array of directions in degrees
        weights: Optional weights (e.g., wind speeds)
    """
    # Convert to radians
    rad = np.deg2rad(directions)
    
    # Calculate weighted mean of sin and cos components
    if weights is not None:
        sin_mean = np.average(np.sin(rad), weights=weights, axis=0)
        cos_mean = np.average(np.cos(rad), weights=weights, axis=0)
    else:
        sin_mean = np.mean(np.sin(rad), axis=0)
        cos_mean = np.mean(np.cos(rad), axis=0)
    
    # Calculate mean direction
    mean_dir = np.arctan2(sin_mean, cos_mean)
    mean_dir_deg = np.rad2deg(mean_dir)
    mean_dir_deg = np.mod(mean_dir_deg, 360)
    
    return mean_dir_deg


def circular_std(directions):
    """Calculate circular standard deviation of wind directions in degrees.
    
    Returns values from 0 (perfect alignment) to ~81.03 degrees (uniform distribution)
    """
    rad = np.deg2rad(directions)
    
    # Calculate mean resultant length
    sin_mean = np.mean(np.sin(rad), axis=0)
    cos_mean = np.mean(np.cos(rad), axis=0)
    R = np.hypot(sin_mean, cos_mean)
    
    # Circular standard deviation
    # Handle R=0 case to avoid log(0)
    R = np.clip(R, 1e-10, 1.0)
    circ_std = np.rad2deg(np.sqrt(-2 * np.log(R)))
    
    return circ_std


def apply_wind_statistic(u_data, v_data, stat_type, stat_param=None):
    """Apply a wind statistic to U and V components along the time dimension.
    
    Args:
        u_data: xarray DataArray of U wind component
        v_data: xarray DataArray of V wind component
        stat_type: Type of statistic to compute
        stat_param: Optional parameter for the statistic (e.g., quantile value)
    
    Returns:
        Result as numpy array
    """
    # Compute wind speed
    speed = compute_wind_speed(u_data.values, v_data.values)
    
    if stat_type == 'mean_speed':
        return np.mean(speed, axis=0)
    
    if stat_type == 'quantile_speed':
        return np.quantile(speed, stat_param, axis=0)
    
    if stat_type == 'max_speed':
        return np.max(speed, axis=0)
    
    if stat_type == 'wind_power':
        # Wind power density is proportional to cube of wind speed
        return np.mean(speed**3, axis=0)
    
    if stat_type == 'calm_freq':
        # Frequency of calm conditions (< 2 m/s, Beaufort 0-1)
        calm_threshold = 2.0
        return np.mean(speed < calm_threshold, axis=0) * 100
    
    if stat_type == 'light_breeze_freq':
        # Frequency of light breeze (> 1.6 m/s, Beaufort 2+)
        light_breeze_threshold = 1.6
        return np.mean(speed > light_breeze_threshold, axis=0) * 100
    
    if stat_type == 'moderate_breeze_freq':
        # Frequency of moderate breeze (> 5.5 m/s, Beaufort 4+)
        moderate_breeze_threshold = 5.5
        return np.mean(speed > moderate_breeze_threshold, axis=0) * 100
    
    if stat_type == 'strong_breeze_freq':
        # Frequency of strong breeze (> 10.8 m/s, Beaufort 6+)
        strong_breeze_threshold = 10.8
        return np.mean(speed > strong_breeze_threshold, axis=0) * 100
    
    if stat_type == 'gale_freq':
        # Frequency of fresh gale (> 17.2 m/s, Beaufort 8+)
        gale_threshold = 17.2
        return np.mean(speed > gale_threshold, axis=0) * 100
    
    if stat_type == 'prevailing_direction':
        # Compute wind directions
        direction = compute_wind_direction(u_data.values, v_data.values)
        # Weight by wind speed for more meaningful prevailing direction
        return circular_mean_direction(direction, weights=speed)
    
    if stat_type == 'direction_variability':
        # Circular standard deviation of wind direction
        direction = compute_wind_direction(u_data.values, v_data.values)
        return circular_std(direction)
    
    if stat_type == 'mean_u':
        return np.mean(u_data.values, axis=0)
    
    if stat_type == 'mean_v':
        return np.mean(v_data.values, axis=0)
    
    raise ValueError(f"Unsupported wind statistic type: {stat_type}")


def plot_wind_stat_map(data, filename, stat_config, label):
    """Plot a single wind statistic map with appropriate styling."""
    
    if stat_config['type'] == 'mean_speed':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Speed [m/s]', vmin=0, vmax=10, cmap='inferno', extend='max'
        )
    elif stat_config['type'] in ['quantile_speed', 'max_speed']:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Speed [m/s]', vmin=0, vmax=30, cmap='inferno', extend='max'
        )
    elif stat_config['type'] == 'wind_power':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Power Density [m³/s³]', vmin=0, vmax=1000, cmap='plasma', extend='max'
        )
    elif stat_config['type'] in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq', 'strong_breeze_freq', 'gale_freq']:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Frequency [%]', vmin=0, vmax=80, cmap='GnBu', extend='max'
        )
    elif stat_config['type'] == 'prevailing_direction':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Direction [degrees from N]', vmin=0, vmax=360, cmap='twilight', extend='neither'
        )
    elif stat_config['type'] == 'direction_variability':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Circular Std Dev [degrees]', vmin=20, vmax=140, cmap='viridis', extend='max'
        )
    elif stat_config['type'] in ['mean_u', 'mean_v']:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Component [m/s]', vmin=-10, vmax=10, cmap='RdBu_r', extend='both'
        )
    else:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Value', vmin=None, vmax=None, cmap='viridis', extend='neither'
        )


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup and config
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting wind statistics generation")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Processing {len(times)} timesteps")

    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    out_root = Path(cfg.generation.io.output_path or './outputs')
    indices = get_channel_indices(dataset)
    
    # Get U and V wind component indices
    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)
    

    # Wind statistic configurations
    WIND_STATISTICS_CONFIG = {
        'mean_speed': {
            'type': 'mean_speed',
            'title': 'Mean Wind Speed'
        },
        'p90_speed': {
            'type': 'quantile_speed',
            'param': 0.90,
            'title': '90th Percentile Wind Speed'
        },
        'max_speed': {
            'type': 'max_speed',
            'title': 'Maximum Wind Speed'
        },
        'wind_power': {
            'type': 'wind_power',
            'title': 'Mean Wind Power Density'
        },
        'calm_freq': {
            'type': 'calm_freq',
            'title': 'Calm Frequency (<2 m/s, Beaufort 0-1)'
        },
        'light_breeze_freq': {
            'type': 'light_breeze_freq',
            'title': 'Light Breeze Frequency (>1.6 m/s, Beaufort 2+)'
        },
        'moderate_breeze_freq': {
            'type': 'moderate_breeze_freq',
            'title': 'Moderate Breeze Frequency (>5.5 m/s, Beaufort 4+)'
        },
        'strong_breeze_freq': {
            'type': 'strong_breeze_freq',
            'title': 'Strong Breeze Frequency (>10.8 m/s, Beaufort 6+)'
        },
        'gale_freq': {
            'type': 'gale_freq',
            'title': 'Gale Frequency (>17.2 m/s, Beaufort 8+)'
        },
        'prevailing_dir': {
            'type': 'prevailing_direction',
            'title': 'Prevailing Wind Direction'
        },
        'dir_variability': {
            'type': 'direction_variability',
            'title': 'Wind Direction Variability'
        },
        'mean_u': {
            'type': 'mean_u',
            'title': 'Mean U-Component'
        },
        'mean_v': {
            'type': 'mean_v',
            'title': 'Mean V-Component'
        }
    }
    
    stat_configs = [
        {
            'stat_name': name,
            'title_stat': config['title'],
            'param': config.get('param'),
            **config
        }
        for name, config in WIND_STATISTICS_CONFIG.items()
    ]

    # Target and baseline modes
    basic_modes = {
        'target': ((u10_out, v10_out), 'COSMO-2 Analysis'),
        'baseline': ((u10_in, v10_in), 'ERA5'),
        'regression-prediction': ((u10_out, v10_out), 'Regression Prediction')
    }
    logger.info(f"Generating {len(stat_configs)} wind statistics for {len(basic_modes)} basic modes + predictions")

    for mode, (wind_channels, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        u_channel, v_channel = wind_channels
        
        # Load all timesteps for this mode
        u_data_list = []
        v_data_list = []
        try:
            for i, ts in enumerate(times):
                if i % LOG_INTERVAL == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False)
                u_data_list.append(data[u_channel])
                v_data_list.append(data[v_channel])
        except Exception as e:
            logger.warning(f"{mode} not available, skipping: {e}")
            continue
        
        # Create xarray DataArrays
        u_mode_data = xr.DataArray(
            np.stack(u_data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        v_mode_data = xr.DataArray(
            np.stack(v_data_list, axis=0),
            dims=['time', 'lat', 'lon'],
            coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
        )
        
        # Compute and plot all statistics for this mode
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            result = apply_wind_statistic(
                u_mode_data, v_mode_data,
                stat_config['type'], stat_config['param']
            )
            
            map_output_dir = out_root / f"maps_wind_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_wind_stat_map(
                result,
                str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'),
                stat_config,
                label
            )

    # Predictions mode: process each member separately to save memory
    logger.info("Processing predictions mode...")
    try:
        data = torch.load(out_root/times[0]/f"{times[0]}-predictions", weights_only=False)
        n_members = data.shape[0]
        logger.info(f"Found {n_members} ensemble members")
        
        for member_idx in range(n_members):
            logger.info(f"Processing prediction member {member_idx+1}/{n_members}")
            
            # Load all timesteps for this member
            u_data_list = []
            v_data_list = []
            for i, ts in enumerate(times):
                if i % LOG_INTERVAL == 0:
                    logger.info(f"Loading prediction member {member_idx} timestep {i+1}/{len(times)}: {ts}")
                pred_data = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False)
                u_data_list.append(pred_data[member_idx, u10_out])
                v_data_list.append(pred_data[member_idx, v10_out])
            
            u_member_data = xr.DataArray(
                np.stack(u_data_list, axis=0),
                dims=['time', 'lat', 'lon'],
                coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
            )
            v_member_data = xr.DataArray(
                np.stack(v_data_list, axis=0),
                dims=['time', 'lat', 'lon'],
                coords={'time': [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]}
            )
            
            # Compute and plot all statistics for this member
            for stat_config in stat_configs:
                logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
                member_result = apply_wind_statistic(
                    u_member_data, v_member_data,
                    stat_config['type'], stat_config['param']
                )
                
                # Create map
                map_output_dir = out_root / f"maps_wind_{stat_config['stat_name']}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
                member_label = f'CorrDiff Member {member_idx+1}'
                plot_wind_stat_map(member_result, member_filename, stat_config, member_label)
    
    except Exception as e:
        logger.warning(f"Predictions not available, skipping: {e}")

    logger.info("All wind statistics maps generated successfully")


if __name__ == '__main__':
    main()
