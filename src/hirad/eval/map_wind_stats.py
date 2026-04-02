import logging
import argparse
import yaml
from pathlib import Path

import hydra
import numpy as np
import torch

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.eval_utils import resolve_times
from hirad.eval.plotting import plot_map, get_channel_indices, GridConfig


def compute_wind_speed(u, v):
    """Compute wind speed from U and V."""
    return np.hypot(u, v)


def compute_wind_direction(u, v, calm_threshold=0.0):
    """Compute wind direction in degrees from N."""
    dir_deg = (np.degrees(np.arctan2(-u, -v)) % 360)
    if calm_threshold > 0:
        speed = np.hypot(u, v)
        dir_deg = np.where(speed <= calm_threshold, np.nan, dir_deg)
    return dir_deg


def apply_all_wind_statistics_streaming(times, out_root, mode, u_channel, v_channel, stat_configs, logger=None, log_interval=100):
    """Compute ALL wind statistics in a single pass through timesteps."""
    accumulators = {}
    counts = {}
    sin_accs = {}
    cos_accs = {}
    speed_accs = {}

    for sc in stat_configs:
        key = sc['stat_name']
        accumulators[key] = None
        counts[key] = 0
        sin_accs[key] = None
        cos_accs[key] = None
        speed_accs[key] = None

    for i, ts in enumerate(times):
        if logger and i % log_interval == 0:
            logger.info(f"  Streaming {mode} timestep {i+1}/{len(times)}: {ts}")

        data = torch.load(out_root / ts / f"{ts}-{mode}", weights_only=False)
        u = data[u_channel].cpu().numpy() if torch.is_tensor(data[u_channel]) else data[u_channel]
        v = data[v_channel].cpu().numpy() if torch.is_tensor(data[v_channel]) else data[v_channel]
        del data

        # Pre-compute shared quantities once per timestep
        speed = compute_wind_speed(u, v)
        direction_calm = None  # lazy
        direction_raw = None   # lazy

        for sc in stat_configs:
            key = sc['stat_name']
            stype = sc['type']

            if stype == 'mean_speed':
                if accumulators[key] is None:
                    accumulators[key] = np.zeros_like(speed)
                accumulators[key] += speed
            elif stype == 'max_speed':
                if accumulators[key] is None:
                    accumulators[key] = np.full_like(speed, -np.inf)
                np.maximum(accumulators[key], speed, out=accumulators[key])
            elif stype == 'wind_power':
                if accumulators[key] is None:
                    accumulators[key] = np.zeros_like(speed)
                accumulators[key] += speed ** 3
            elif stype == 'mean_u':
                if accumulators[key] is None:
                    accumulators[key] = np.zeros_like(u)
                accumulators[key] += u
            elif stype == 'mean_v':
                if accumulators[key] is None:
                    accumulators[key] = np.zeros_like(v)
                accumulators[key] += v
            elif stype in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq',
                           'strong_breeze_freq', 'gale_freq']:
                thresholds = {
                    'calm_freq': 2.0, 'light_breeze_freq': 1.6,
                    'moderate_breeze_freq': 5.5, 'strong_breeze_freq': 10.8,
                    'gale_freq': 17.2
                }
                threshold = thresholds[stype]
                if accumulators[key] is None:
                    accumulators[key] = np.zeros_like(speed)
                if stype == 'calm_freq':
                    accumulators[key] += (speed < threshold).astype(float)
                else:
                    accumulators[key] += (speed > threshold).astype(float)
            elif stype == 'prevailing_direction':
                if direction_calm is None:
                    direction_calm = compute_wind_direction(u, v, calm_threshold=1.0)
                rad = np.deg2rad(direction_calm)
                weighted_sin = np.sin(rad) * speed
                weighted_cos = np.cos(rad) * speed
                if sin_accs[key] is None:
                    sin_accs[key] = np.zeros_like(weighted_sin)
                    cos_accs[key] = np.zeros_like(weighted_cos)
                    speed_accs[key] = np.zeros_like(speed)
                sin_accs[key] += np.nan_to_num(weighted_sin, 0)
                cos_accs[key] += np.nan_to_num(weighted_cos, 0)
                speed_accs[key] += speed
            elif stype == 'direction_variability':
                if direction_raw is None:
                    direction_raw = compute_wind_direction(u, v)
                rad = np.deg2rad(direction_raw)
                if sin_accs[key] is None:
                    sin_accs[key] = np.zeros_like(speed)
                    cos_accs[key] = np.zeros_like(speed)
                sin_accs[key] += np.sin(rad)
                cos_accs[key] += np.cos(rad)

            counts[key] += 1

        del u, v, speed, direction_calm, direction_raw

    # Finalize all statistics
    results = {}
    for sc in stat_configs:
        key = sc['stat_name']
        stype = sc['type']
        count = counts[key]

        if stype == 'prevailing_direction':
            mean_dir = np.arctan2(
                sin_accs[key] / (speed_accs[key] + 1e-10),
                cos_accs[key] / (speed_accs[key] + 1e-10)
            )
            results[key] = np.mod(np.rad2deg(mean_dir), 360)
        elif stype == 'direction_variability':
            R = np.clip(np.hypot(sin_accs[key] / count, cos_accs[key] / count), 1e-10, 1.0)
            results[key] = np.rad2deg(np.sqrt(-2 * np.log(R)))
        elif stype == 'max_speed':
            results[key] = accumulators[key]
        elif stype in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq',
                       'strong_breeze_freq', 'gale_freq']:
            results[key] = (accumulators[key] / count) * 100
        else:
            results[key] = accumulators[key] / count

    return results


def plot_wind_stat_map(data, filename, stat_config, label, grid_cfg):
    """Plot wind statistic map."""
    if stat_config['type'] == 'mean_speed':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Speed [m/s]', vmin=0, vmax=10, cmap='inferno', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'max_speed':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Speed [m/s]', vmin=0, vmax=30, cmap='inferno', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'wind_power':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Power Density [m³/s³]', vmin=0, vmax=1000, cmap='plasma', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq', 'strong_breeze_freq', 'gale_freq']:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Frequency [%]', vmin=0, vmax=80, cmap='GnBu', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'prevailing_direction':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Direction [degrees from N]', vmin=0, vmax=360, cmap='twilight', extend='neither', grid_cfg=grid_cfg
        )
    elif stat_config['type'] == 'direction_variability':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Circular Std Dev [degrees]', vmin=20, vmax=140, cmap='viridis', extend='max', grid_cfg=grid_cfg
        )
    elif stat_config['type'] in ['mean_u', 'mean_v']:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Wind Component [m/s]', vmin=-5, vmax=5, cmap='RdBu_r', extend='both', grid_cfg=grid_cfg
        )
    else:
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]}',
            label='Value', vmin=None, vmax=None, cmap='viridis', extend='neither', grid_cfg=grid_cfg
        )


def main(cfg: dict):
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

    logger.info("Starting wind statistics generation")
    times = resolve_times(cfg, gen_cfg)
    if times is None:
        logger.error("No times, times_range, or times_ranges specified in config or generation config.")
        return
    logger.info(f"Processing {len(times)} timesteps")

    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.get("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    indices = get_channel_indices(dataset)
    
    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)


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

    basic_modes = {
        'target': ((u10_out, v10_out), 'Target'),
        'baseline': ((u10_in, v10_in), 'Input'),
        'regression-prediction': ((u10_out, v10_out), 'Regression Prediction')
    }

    logger.info(f"Generating {len(stat_configs)} statistics for {len(basic_modes)} modes + predictions")

    log_interval = cfg.get("log_interval", 100)

    for mode, (wind_channels, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        u_channel, v_channel = wind_channels
        
        try:
            test_data = torch.load(out_root/times[0]/f"{times[0]}-{mode}", weights_only=False)
            del test_data
        except Exception as e:
            logger.warning(f"{mode} not available: {e}")
            continue
        
        try:
            results = apply_all_wind_statistics_streaming(
                times, out_root, mode, u_channel, v_channel,
                stat_configs, logger=logger, log_interval=log_interval
            )
            
            for stat_config in stat_configs:
                key = stat_config['stat_name']
                map_output_dir = output_path / f"maps_wind_{key}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                plot_wind_stat_map(
                    results[key],
                    str(map_output_dir / f'{mode}_{key}'),
                    stat_config,
                    label,
                    grid_cfg
                )
            del results
        except Exception as e:
            logger.error(f"Failed computing statistics for {mode}: {e}")
            continue


    logger.info("Processing predictions mode...")
    try:
        data = torch.load(out_root/times[0]/f"{times[0]}-predictions", weights_only=False)
        n_members = data.shape[0]
        del data
        logger.info(f"Found {n_members} ensemble members")
        
        # Initialize accumulators for all members × all statistics at once
        # Each accumulator is keyed by (member_idx, stat_name)
        accumulators = {}
        counts = {}
        sin_accs = {}
        cos_accs = {}
        speed_accs = {}
        
        for member_idx in range(n_members):
            for stat_config in stat_configs:
                key = (member_idx, stat_config['stat_name'])
                accumulators[key] = None
                counts[key] = 0
                sin_accs[key] = None
                cos_accs[key] = None
                speed_accs[key] = None
        
        # Single pass over timesteps — load each file once for all members
        for i, ts in enumerate(times):
            if i % log_interval == 0:
                logger.info(f"Loading predictions timestep {i+1}/{len(times)}: {ts}")
            
            pred_data = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False)
            
            for member_idx in range(n_members):
                u_data = pred_data[member_idx, u10_out]
                v_data = pred_data[member_idx, v10_out]
                u = u_data.cpu().numpy() if torch.is_tensor(u_data) else u_data
                v = v_data.cpu().numpy() if torch.is_tensor(v_data) else v_data
                
                # Pre-compute shared quantities once per member per timestep
                speed = compute_wind_speed(u, v)
                direction_calm = None
                direction_raw = None
                
                for stat_config in stat_configs:
                    key = (member_idx, stat_config['stat_name'])
                    stype = stat_config['type']
                    
                    if stype == 'mean_speed':
                        if accumulators[key] is None:
                            accumulators[key] = np.zeros_like(speed)
                        accumulators[key] += speed
                    elif stype == 'max_speed':
                        if accumulators[key] is None:
                            accumulators[key] = np.full_like(speed, -np.inf)
                        np.maximum(accumulators[key], speed, out=accumulators[key])
                    elif stype == 'wind_power':
                        if accumulators[key] is None:
                            accumulators[key] = np.zeros_like(speed)
                        accumulators[key] += speed ** 3
                    elif stype == 'mean_u':
                        if accumulators[key] is None:
                            accumulators[key] = np.zeros_like(u)
                        accumulators[key] += u
                    elif stype == 'mean_v':
                        if accumulators[key] is None:
                            accumulators[key] = np.zeros_like(v)
                        accumulators[key] += v
                    elif stype in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq',
                                   'strong_breeze_freq', 'gale_freq']:
                        thresholds = {
                            'calm_freq': 2.0, 'light_breeze_freq': 1.6,
                            'moderate_breeze_freq': 5.5, 'strong_breeze_freq': 10.8,
                            'gale_freq': 17.2
                        }
                        threshold = thresholds[stype]
                        if accumulators[key] is None:
                            accumulators[key] = np.zeros_like(speed)
                        if stype == 'calm_freq':
                            accumulators[key] += (speed < threshold).astype(float)
                        else:
                            accumulators[key] += (speed > threshold).astype(float)
                    elif stype == 'prevailing_direction':
                        if direction_calm is None:
                            direction_calm = compute_wind_direction(u, v, calm_threshold=1.0)
                        rad = np.deg2rad(direction_calm)
                        weighted_sin = np.sin(rad) * speed
                        weighted_cos = np.cos(rad) * speed
                        if sin_accs[key] is None:
                            sin_accs[key] = np.zeros_like(weighted_sin)
                            cos_accs[key] = np.zeros_like(weighted_cos)
                            speed_accs[key] = np.zeros_like(speed)
                        sin_accs[key] += np.nan_to_num(weighted_sin, 0)
                        cos_accs[key] += np.nan_to_num(weighted_cos, 0)
                        speed_accs[key] += speed
                    elif stype == 'direction_variability':
                        if direction_raw is None:
                            direction_raw = compute_wind_direction(u, v)
                        rad = np.deg2rad(direction_raw)
                        if sin_accs[key] is None:
                            sin_accs[key] = np.zeros_like(speed)
                            cos_accs[key] = np.zeros_like(speed)
                        sin_accs[key] += np.sin(rad)
                        cos_accs[key] += np.cos(rad)
                    
                    counts[key] += 1
                
                del u, v, speed, direction_calm, direction_raw
            
            del pred_data
        
        # Finalize and plot all statistics for all members
        for member_idx in range(n_members):
            logger.info(f"Finalizing and plotting member {member_idx+1}/{n_members}")
            for stat_config in stat_configs:
                stat_key = stat_config['stat_name']
                key = (member_idx, stat_key)
                stype = stat_config['type']
                count = counts[key]
                
                try:
                    if stype == 'prevailing_direction':
                        mean_dir = np.arctan2(
                            sin_accs[key] / (speed_accs[key] + 1e-10),
                            cos_accs[key] / (speed_accs[key] + 1e-10)
                        )
                        member_result = np.mod(np.rad2deg(mean_dir), 360)
                    elif stype == 'direction_variability':
                        R = np.clip(np.hypot(sin_accs[key] / count, cos_accs[key] / count), 1e-10, 1.0)
                        member_result = np.rad2deg(np.sqrt(-2 * np.log(R)))
                    elif stype == 'max_speed':
                        member_result = accumulators[key]
                    elif stype in ['calm_freq', 'light_breeze_freq', 'moderate_breeze_freq',
                                   'strong_breeze_freq', 'gale_freq']:
                        member_result = (accumulators[key] / count) * 100
                    else:
                        member_result = accumulators[key] / count
                    
                    map_output_dir = output_path / f"maps_wind_{stat_key}"
                    map_output_dir.mkdir(parents=True, exist_ok=True)
                    member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_key}')
                    plot_wind_stat_map(member_result, member_filename, stat_config, f'CorrDiff Member {member_idx+1}', grid_cfg)
                    del member_result
                except Exception as e:
                    logger.error(f"Failed {stat_config['title_stat']} for member {member_idx+1}: {e}")
                    continue
    
    except Exception as e:
        logger.warning(f"Predictions not available: {e}")

    logger.info("Wind statistics generation complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)