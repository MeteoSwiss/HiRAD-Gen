import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import plot_map, get_channel_indices, LOG_INTERVAL


def compute_wind_speed(u, v):
    """Compute wind speed from U and V components."""
    return np.hypot(u, v)


def compute_wind_direction(u, v, calm_threshold=0.0):
    dir_deg = (np.degrees(np.arctan2(-u, -v)) % 360)
    if calm_threshold > 0:
        speed = np.hypot(u, v)
        dir_deg = np.where(speed <= calm_threshold, np.nan, dir_deg)
    return dir_deg


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


def apply_wind_statistic_streaming(times, out_root, mode, u_channel, v_channel, stat_type, stat_param=None):
    """Apply a wind statistic by streaming through timesteps to minimize memory usage.
    
    Args:
        times: List of timestep strings
        out_root: Path to output directory
        mode: Mode name (e.g., 'target', 'baseline', 'predictions')
        u_channel: Index of U wind component
        v_channel: Index of V wind component
        stat_type: Type of statistic to compute
        stat_param: Optional parameter for the statistic (e.g., quantile value)
    
    Returns:
        Result as numpy array
    """
    # All statistics can now be computed incrementally
    # Initialize accumulators
    accumulator = None
    count = 0
    
    # Special accumulators for circular statistics
    if stat_type == 'prevailing_direction':
        sin_acc = None
        cos_acc = None
        speed_acc = None
    elif stat_type == 'direction_variability':
        sin_acc = None
        cos_acc = None
    
    for ts in times:
        if mode == 'predictions':
            data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False)
            u = data[u_channel].cpu().numpy()
            v = data[v_channel].cpu().numpy()
        else:
            data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False)
            u = data[u_channel].cpu().numpy() if torch.is_tensor(data[u_channel]) else data[u_channel]
            v = data[v_channel].cpu().numpy() if torch.is_tensor(data[v_channel]) else data[v_channel]
        
        if stat_type == 'mean_speed':
            speed = compute_wind_speed(u, v)
            if accumulator is None:
                accumulator = np.zeros_like(speed)
            accumulator += speed
        elif stat_type == 'max_speed':
            speed = compute_wind_speed(u, v)
            if accumulator is None:
                accumulator = np.full_like(speed, -np.inf)
            accumulator = np.maximum(accumulator, speed)
        elif stat_type == 'wind_power':
            speed = compute_wind_speed(u, v)
            if accumulator is None:
                accumulator = np.zeros_like(speed)
            accumulator += speed**3
        elif stat_type == 'mean_u':
            if accumulator is None:
                accumulator = np.zeros_like(u)
            accumulator += u
        elif stat_type == 'mean_v':
            if accumulator is None:
                accumulator = np.zeros_like(v)
            accumulator += v
        elif stat_type in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq', 
                          'strong_breeze_freq', 'gale_freq']:
            speed = compute_wind_speed(u, v)
            thresholds = {
                'calm_freq': 2.0,
                'light_breeze_freq': 1.6,
                'moderate_breeze_freq': 5.5,
                'strong_breeze_freq': 10.8,
                'gale_freq': 17.2
            }
            threshold = thresholds[stat_type]
            if accumulator is None:
                accumulator = np.zeros_like(speed)
            if stat_type == 'calm_freq':
                accumulator += (speed < threshold).astype(float)
            else:
                accumulator += (speed > threshold).astype(float)
        elif stat_type == 'prevailing_direction':
            speed = compute_wind_speed(u, v)
            direction = compute_wind_direction(u, v, calm_threshold=1.0)
            rad = np.deg2rad(direction)
            weighted_sin = np.sin(rad) * speed
            weighted_cos = np.cos(rad) * speed
            
            if sin_acc is None:
                sin_acc = np.zeros_like(weighted_sin)
                cos_acc = np.zeros_like(weighted_cos)
                speed_acc = np.zeros_like(speed)
            
            sin_acc += np.nan_to_num(weighted_sin, 0)
            cos_acc += np.nan_to_num(weighted_cos, 0)
            speed_acc += speed
        elif stat_type == 'direction_variability':
            direction = compute_wind_direction(u, v)
            rad = np.deg2rad(direction)
            
            if sin_acc is None:
                sin_acc = np.zeros_like(np.sin(rad))
                cos_acc = np.zeros_like(np.cos(rad))
            
            sin_acc += np.sin(rad)
            cos_acc += np.cos(rad)
        
        count += 1
        del data, u, v
    
    # Finalize computation
    if stat_type == 'prevailing_direction':
        mean_sin = sin_acc / (speed_acc + 1e-10)
        mean_cos = cos_acc / (speed_acc + 1e-10)
        mean_dir = np.arctan2(mean_sin, mean_cos)
        return np.mod(np.rad2deg(mean_dir), 360)
    elif stat_type == 'direction_variability':
        # Calculate mean resultant length
        sin_mean = sin_acc / count
        cos_mean = cos_acc / count
        R = np.hypot(sin_mean, cos_mean)
        # Circular standard deviation
        R = np.clip(R, 1e-10, 1.0)
        return np.rad2deg(np.sqrt(-2 * np.log(R)))
    elif stat_type == 'max_speed':
        return accumulator
    elif stat_type in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq', 
                      'strong_breeze_freq', 'gale_freq']:
        return (accumulator / count) * 100
    else:
        return accumulator / count


def plot_wind_stat_map(data, filename, stat_config, label):
    """Plot a single wind statistic map with appropriate styling."""
    
    if stat_config['type'] == 'mean_speed':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Speed [m/s]', vmin=0, vmax=10, cmap='inferno', extend='max'
        )
    elif stat_config['type'] == 'max_speed':
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
            label='Wind Component [m/s]', vmin=-5, vmax=5, cmap='RdBu_r', extend='both'
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
        
        # Check if mode is available
        try:
            test_data = torch.load(out_root/times[0]/f"{times[0]}-{mode}", weights_only=False)
            del test_data
        except Exception as e:
            logger.warning(f"{mode} not available, skipping: {e}")
            continue
        
        # Compute and plot all statistics for this mode using streaming approach
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            try:
                result = apply_wind_statistic_streaming(
                    times, out_root, mode, u_channel, v_channel,
                    stat_config['type'], stat_config.get('param')
                )
                
                map_output_dir = out_root / f"maps_wind_{stat_config['stat_name']}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                plot_wind_stat_map(
                    result,
                    str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'),
                    stat_config,
                    label
                )
                del result
            except Exception as e:
                logger.error(f"Failed to compute {stat_config['title_stat']} for {mode}: {e}")
                continue

    # Predictions mode: process each member and statistic separately to save memory
    logger.info("Processing predictions mode...")
    try:
        data = torch.load(out_root/times[0]/f"{times[0]}-predictions", weights_only=False)
        n_members = data.shape[0]
        del data
        logger.info(f"Found {n_members} ensemble members")
        
        for member_idx in range(n_members):
            logger.info(f"Processing prediction member {member_idx+1}/{n_members}")
            
            # Process each statistic for this member using streaming
            for stat_config in stat_configs:
                logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
                try:
                    # Create a custom mode string for this member
                    def load_member_data(ts):
                        pred_data = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False)
                        u = pred_data[member_idx, u10_out].cpu().numpy()
                        v = pred_data[member_idx, v10_out].cpu().numpy()
                        del pred_data
                        return u, v
                    
                    # All statistics are now streaming
                    accumulator = None
                    count = 0
                    
                    # Special accumulators for circular statistics
                    if stat_config['type'] == 'prevailing_direction':
                        sin_acc = None
                        cos_acc = None
                        speed_acc = None
                    elif stat_config['type'] == 'direction_variability':
                        sin_acc = None
                        cos_acc = None
                    
                    for i, ts in enumerate(times):
                        if i % LOG_INTERVAL == 0:
                            logger.info(f"Loading prediction member {member_idx} timestep {i+1}/{len(times)}: {ts}")
                        
                        u, v = load_member_data(ts)
                        
                        if stat_config['type'] == 'mean_speed':
                            speed = compute_wind_speed(u, v)
                            if accumulator is None:
                                accumulator = np.zeros_like(speed)
                            accumulator += speed
                        elif stat_config['type'] == 'max_speed':
                            speed = compute_wind_speed(u, v)
                            if accumulator is None:
                                accumulator = np.full_like(speed, -np.inf)
                            accumulator = np.maximum(accumulator, speed)
                        elif stat_config['type'] == 'wind_power':
                            speed = compute_wind_speed(u, v)
                            if accumulator is None:
                                accumulator = np.zeros_like(speed)
                            accumulator += speed**3
                        elif stat_config['type'] == 'mean_u':
                            if accumulator is None:
                                accumulator = np.zeros_like(u)
                            accumulator += u
                        elif stat_config['type'] == 'mean_v':
                            if accumulator is None:
                                accumulator = np.zeros_like(v)
                            accumulator += v
                        elif stat_config['type'] in ['calm_freq', 'light_breeze_freq', 
                                                     'moderate_breeze_freq', 'strong_breeze_freq', 
                                                     'gale_freq']:
                            speed = compute_wind_speed(u, v)
                            thresholds = {
                                'calm_freq': 2.0,
                                'light_breeze_freq': 1.6,
                                'moderate_breeze_freq': 5.5,
                                'strong_breeze_freq': 10.8,
                                'gale_freq': 17.2
                            }
                            threshold = thresholds[stat_config['type']]
                            if accumulator is None:
                                accumulator = np.zeros_like(speed)
                            if stat_config['type'] == 'calm_freq':
                                accumulator += (speed < threshold).astype(float)
                            else:
                                accumulator += (speed > threshold).astype(float)
                        elif stat_config['type'] == 'prevailing_direction':
                            speed = compute_wind_speed(u, v)
                            direction = compute_wind_direction(u, v, calm_threshold=1.0)
                            rad = np.deg2rad(direction)
                            weighted_sin = np.sin(rad) * speed
                            weighted_cos = np.cos(rad) * speed
                            
                            if sin_acc is None:
                                sin_acc = np.zeros_like(weighted_sin)
                                cos_acc = np.zeros_like(weighted_cos)
                                speed_acc = np.zeros_like(speed)
                            
                            sin_acc += np.nan_to_num(weighted_sin, 0)
                            cos_acc += np.nan_to_num(weighted_cos, 0)
                            speed_acc += speed
                        elif stat_config['type'] == 'direction_variability':
                            direction = compute_wind_direction(u, v)
                            rad = np.deg2rad(direction)
                            
                            if sin_acc is None:
                                sin_acc = np.zeros_like(np.sin(rad))
                                cos_acc = np.zeros_like(np.cos(rad))
                            
                            sin_acc += np.sin(rad)
                            cos_acc += np.cos(rad)
                        
                        count += 1
                        del u, v
                    
                    # Finalize computation
                    if stat_config['type'] == 'prevailing_direction':
                        mean_sin = sin_acc / (speed_acc + 1e-10)
                        mean_cos = cos_acc / (speed_acc + 1e-10)
                        mean_dir = np.arctan2(mean_sin, mean_cos)
                        member_result = np.mod(np.rad2deg(mean_dir), 360)
                    elif stat_config['type'] == 'direction_variability':
                        # Calculate mean resultant length
                        sin_mean = sin_acc / count
                        cos_mean = cos_acc / count
                        R = np.hypot(sin_mean, cos_mean)
                        # Circular standard deviation
                        R = np.clip(R, 1e-10, 1.0)
                        member_result = np.rad2deg(np.sqrt(-2 * np.log(R)))
                    elif stat_config['type'] == 'max_speed':
                        member_result = accumulator
                    elif stat_config['type'] in ['calm_freq', 'light_breeze_freq', 
                                                 'moderate_breeze_freq', 'strong_breeze_freq', 
                                                 'gale_freq']:
                        member_result = (accumulator / count) * 100
                    else:
                        member_result = accumulator / count
                    
                    # Create map
                    map_output_dir = out_root / f"maps_wind_{stat_config['stat_name']}"
                    map_output_dir.mkdir(parents=True, exist_ok=True)
                    member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
                    member_label = f'CorrDiff Member {member_idx+1}'
                    plot_wind_stat_map(member_result, member_filename, stat_config, member_label)
                    del member_result
                
                except Exception as e:
                    logger.error(f"Failed to compute {stat_config['title_stat']} for member {member_idx+1}: {e}")
                    continue
    
    except Exception as e:
        logger.warning(f"Predictions not available, skipping: {e}")

    logger.info("All wind statistics maps generated successfully")


if __name__ == '__main__':
    main()
