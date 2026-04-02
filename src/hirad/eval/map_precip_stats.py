import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import numba

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.eval_utils import resolve_times
from hirad.eval.plotting import (
    plot_map_precipitation, plot_map, get_channel_indices, GridConfig
)


@numba.njit
def _longest_spell(x):
    """Longest consecutive run of True values in a 1-D boolean array."""
    best = 0
    cur = 0
    for i in range(x.shape[0]):
        if x[i]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


@numba.njit(parallel=True)
def _consecutive_spell_2d(condition_3d):
    """
    condition_3d: bool array of shape (T, H, W).
    Returns int array of shape (H, W) with longest spell per grid point.
    """
    T, H, W = condition_3d.shape
    out = np.empty((H, W), dtype=np.int64)
    for i in numba.prange(H):
        for j in range(W):
            out[i, j] = _longest_spell(condition_3d[:, i, j])
    return out


def consecutive_spell(data_np, condition_fn):
    """
    data_np: numpy array (T, H, W)
    condition_fn: callable that takes the array and returns bool array of same shape
    """
    cond = condition_fn(data_np)
    return _consecutive_spell_2d(cond)


def apply_statistic(data_np, times_dt, stat_type, stat_param, wet_threshold=0.1):
    """
    Apply statistic on array containing time sequence of total precipitation map.
    data_np: (T, H, W) float array
    times_dt: list of datetime objects (length T)
    Returns: (H, W) numpy array
    """
    if stat_type == 'mean':
        return np.mean(data_np, axis=0)

    if stat_type == 'quantile':
        return np.quantile(data_np, stat_param, axis=0)

    if stat_type == 'Rx1hr':
        return np.max(data_np, axis=0)

    # For daily aggregations, build daily sums using xarray (fast groupby)
    if stat_type in ('Rx1day', 'Rx5day', 'cdd', 'cwd'):
        da = xr.DataArray(
            data_np, dims=['time', 'lat', 'lon'],
            coords={'time': times_dt}
        )
        daily = da.resample(time="1D").sum("time").values  # (D, H, W)

        if stat_type == 'Rx1day':
            return np.max(daily, axis=0)

        if stat_type == 'Rx5day':
            # Rolling sum along time axis using a cumsum trick
            D, H, W = daily.shape
            if D < 5:
                return np.sum(daily, axis=0)
            rolling5 = np.empty((D - 4, H, W), dtype=daily.dtype)
            for t in range(D - 4):
                rolling5[t] = daily[t:t+5].sum(axis=0)
            return np.max(rolling5, axis=0)

        if stat_type == 'cdd':
            return consecutive_spell(daily, lambda x: x < 1.0)

        if stat_type == 'cwd':
            return consecutive_spell(daily, lambda x: x >= 1.0)

    if stat_type == 'weth_freq':
        return np.mean(data_np / 24.0 > wet_threshold, axis=0) * 100.0

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


def _load_predictions_all_members(filepath, conv_factor):
    """Load prediction file once and return (n_members, C, H, W) tensor."""
    return torch.load(filepath, weights_only=False) * conv_factor


def main(cfg: dict):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    grid_cfg = GridConfig(
        lat=np.arange(cfg.get("lat_start"), cfg.get("lat_end") + cfg.get("lat_step"), cfg.get("lat_step")),
        lon=np.arange(cfg.get("lon_start"), cfg.get("lon_end") + cfg.get("lon_step"), cfg.get("lon_step")),
        height=cfg.get("height"),
        width=cfg.get("width"),
        relax_zone=cfg.get("relax_zone")
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
    times = resolve_times(cfg, gen_cfg)
    if times is None:
        logger.error("No times, times_range, or times_ranges specified in config or generation config.")
        return
    logger.info(f"Processing {len(times)} timesteps")

    times_dt = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]

    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.get("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    conv_factor = cfg.get("conv_factor")
    log_interval = cfg.get("log_interval", 100)
    wet_threshold = cfg.get("wet_threshold", 0.1)

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
        {'stat_name': name, 'title_stat': config['title'], 'param': config.get('param'), **config}
        for name, config in STATISTICS_CONFIG.items()
    ]

    # --- Basic modes: target, baseline, regression-prediction ---
    basic_modes = {
        'target': (tp_out, 'Target'),
        'baseline': (tp_in, 'Input'),
        'regression-prediction': (tp_out, 'Regression Prediction')
    }

    for mode, (tp_channel, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        data_list = []
        try:
            for i, ts in enumerate(times):
                if i % log_interval == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                data = torch.load(out_root / ts / f"{ts}-{mode}", weights_only=False) * conv_factor
                data_list.append(data[tp_channel].numpy() if isinstance(data, torch.Tensor) else data[tp_channel])
        except Exception:
            logger.warning(f"{mode} not available, skipping")
            continue

        # Stack into (T, H, W) numpy array
        mode_data = np.stack(data_list, axis=0).astype(np.float64)
        del data_list

        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            result = apply_statistic(mode_data, times_dt, stat_config['type'], stat_config['param'], wet_threshold)
            map_output_dir = output_path / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_stat_map(result, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label, grid_cfg)

    # --- Predictions: load each file ONCE, distribute to all members ---
    logger.info("Processing predictions mode...")
    sample_data = torch.load(out_root / times[0] / f"{times[0]}-predictions", weights_only=False)
    n_members = sample_data.shape[0]
    del sample_data
    logger.info(f"Found {n_members} ensemble members")

    # Pre-allocate arrays for ALL members at once: (n_members, T, H, W)
    # If memory is tight, we can do this in chunks. For 16 members × 2200 × 704 × 1088 × 4 bytes ≈ 107 GB
    # Instead we can do cummulative statistics on the fly without storing all members in memory (like in map_wind_stats), but this works for now.
    H, W = cfg.get("height"), cfg.get("width")
    member_arrays = [np.empty((len(times), H, W), dtype=np.float32) for _ in range(n_members)]

    logger.info("Loading all prediction timesteps (single pass over files)...")
    for i, ts in enumerate(times):
        if i % log_interval == 0:
            logger.info(f"Loading predictions timestep {i+1}/{len(times)}: {ts}")
        pred_data = torch.load(out_root / ts / f"{ts}-predictions", weights_only=False) * conv_factor
        for m in range(n_members):
            member_arrays[m][i] = (pred_data[m, tp_out].numpy() if isinstance(pred_data, torch.Tensor)
                                   else pred_data[m, tp_out])
    del pred_data

    for member_idx in range(n_members):
        logger.info(f"Computing statistics for prediction member {member_idx+1}/{n_members}")
        member_data = member_arrays[member_idx].astype(np.float64)

        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
            member_result = apply_statistic(member_data, times_dt, stat_config['type'], stat_config['param'], wet_threshold)
            map_output_dir = output_path / f"maps_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'CorrDiff Member {member_idx+1}'
            plot_stat_map(member_result, member_filename, stat_config, member_label, grid_cfg)

    del member_arrays
    logger.info("All precipitation statistics maps generated successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)