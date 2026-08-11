import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import numba

from hirad.eval.eval_utils import get_channel_indices, grid_cfg_from_cfg, load_generation_setup, parse_eval_cli, precip_conv_factor, precip_unit_label, sample_interval_hours, resolve_ts_dir
from hirad.eval.plotting import (
    plot_difference_map, plot_map, plot_map_precipitation
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

    if stat_type == 'Rx_step':
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
        return np.mean(data_np > wet_threshold, axis=0) * 100.0

    raise ValueError(f"Unsupported statistic type: {stat_type}")


def plot_stat_map(data, filename, stat_config, label, grid_cfg, unit='mm/h'):
    """Plot a single statistic map with appropriate styling."""
    if stat_config['type'] == 'weth_freq':
        plot_map(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} (%)',
            label='Wet-Period Frequency [%]', vmin=0, vmax=30, cmap='PuBu', extend='max', grid_cfg=grid_cfg
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
        stat_unit = {'Rx1day': 'mm/day', 'Rx5day': 'mm'}.get(stat_config['type'], unit)
        plot_map_precipitation(
            data, filename,
            title=f'{label}: {stat_config["title_stat"]} Precipitation',
            threshold=stat_config['threshold'], rfac=1.0, grid_cfg=grid_cfg, label=stat_unit,
        )


def _difference_label(stat_type, unit='mm/h'):
    if stat_type == 'weth_freq':
        return 'Difference [%]'
    if stat_type in ('cdd', 'cwd'):
        return 'Difference [days]'
    if stat_type == 'Rx1day':
        return 'Difference [mm/day]'
    if stat_type == 'Rx5day':
        return 'Difference [mm]'
    return f'Difference [{unit}]'


def main(cfg: dict):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    grid_cfg = grid_cfg_from_cfg(cfg)

    logger.info("Starting precipitation statistics generation")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Processing {len(times)} timesteps")

    unit = precip_unit_label(times)
    hours = sample_interval_hours(times)
    rx_key = 'Rx1hr' if hours == 1 else f'Rx{hours}hr'
    logger.info(f"Precipitation unit: {unit}; single-step max statistic: {rx_key}")

    times_dt = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    conv_factor = precip_conv_factor(cfg)
    log_interval = cfg.get("log_interval", 100)
    wet_threshold = cfg.get("wet_threshold", 0.1)

    STATISTICS_CONFIG = {
        'mean': {'type': 'mean', 'threshold': 0.01, 'title': 'Mean'},
        'p99': {'type': 'quantile', 'param': 0.99, 'threshold': 0.1, 'title': '99th Percentile'},
        'p99.9': {'type': 'quantile', 'param': 0.999, 'threshold': 0.1, 'title': '99.9th Percentile'},
        'p99.99': {'type': 'quantile', 'param': 0.9999, 'threshold': 0.1, 'title': '99.99th Percentile'},
        rx_key: {'type': 'Rx_step', 'threshold': 0.1, 'title': f'Maximum Single-Step Amount ({rx_key})'},
        'Rx1day': {'type': 'Rx1day', 'threshold': 0.1, 'title': 'Maximum 1-day Amount (Rx1day)'},
        'Rx5day': {'type': 'Rx5day', 'threshold': 0.1, 'title': 'Maximum 5-day Total (Rx5day)'},
        'cdd': {'type': 'cdd', 'threshold': 0.1, 'title': 'Consecutive Dry Days (CDD)'},
        'cwd': {'type': 'cwd', 'threshold': 0.1, 'title': 'Consecutive Wet Days (CWD)'},
        'weth_freq': {'type': 'weth_freq', 'threshold': 0.01, 'title': 'Wet-Period Frequency'}
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

    mode_results = {}

    for mode, (tp_channel, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        data_list = []
        try:
            for i, ts in enumerate(times):
                if i % log_interval == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                data = torch.load(resolve_ts_dir(out_root, ts) / ts / f"{ts}-{mode}", weights_only=False) * conv_factor
                data_list.append(data[tp_channel].numpy() if isinstance(data, torch.Tensor) else data[tp_channel])
        except Exception:
            logger.warning(f"{mode} not available, skipping")
            continue

        # Stack into (T, H, W) numpy array. float32 to save memory.
        mode_data = np.stack(data_list, axis=0).astype(np.float32)
        del data_list

        mode_results[mode] = {}
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            result = apply_statistic(mode_data, times_dt, stat_config['type'], stat_config['param'], wet_threshold)
            mode_results[mode][stat_config['stat_name']] = result
            map_output_dir = output_path / f"maps_precip_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_stat_map(result, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label, grid_cfg, unit=unit)

        del mode_data

    target_results = mode_results.get('target')
    if target_results is None:
        logger.warning("Target mode not available; skipping prediction-minus-target difference maps for basic modes")
    else:
        for mode, (_, label) in basic_modes.items():
            if mode == 'target' or mode not in mode_results:
                continue
            logger.info(f"Generating {mode} minus target difference maps")
            for stat_config in stat_configs:
                stat_name = stat_config['stat_name']
                if stat_name not in mode_results[mode] or stat_name not in target_results:
                    continue
                diff = mode_results[mode][stat_name] - target_results[stat_name]
                map_output_dir = output_path / f"maps_precip_{stat_name}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                plot_difference_map(
                    diff,
                    str(map_output_dir / f'{mode}_minus_target_{stat_name}'),
                    title=f'{label} - Target: {stat_config["title_stat"]} Difference',
                    label=_difference_label(stat_config['type'], unit),
                    grid_cfg=grid_cfg,
                )

    # --- Predictions: process ONE member at a time to bound memory usage ---
    logger.info("Processing predictions mode...")
    sample_data = torch.load(resolve_ts_dir(out_root, times[0]) / times[0] / f"{times[0]}-predictions", weights_only=False)
    n_members = sample_data.shape[0]
    del sample_data
    logger.info(f"Found {n_members} ensemble members")

    H: int = cfg["height"]
    W: int = cfg["width"]
    member_data = np.empty((len(times), H, W), dtype=np.float32)
    has_target_for_diff = target_results is not None
    if not has_target_for_diff:
        logger.warning("Target mode not available; skipping prediction-minus-target difference maps for members")

    for member_idx in range(n_members):
        logger.info(f"Loading prediction member {member_idx+1}/{n_members} (single pass over files)...")
        for i, ts in enumerate(times):
            if i % log_interval == 0:
                logger.info(f"Loading predictions member {member_idx+1} timestep {i+1}/{len(times)}: {ts}")
            pred_data = torch.load(resolve_ts_dir(out_root, ts) / ts / f"{ts}-predictions", weights_only=False)
            member_slice = pred_data[member_idx, tp_out]
            member_data[i] = (member_slice.numpy() if isinstance(member_slice, torch.Tensor) else member_slice) * conv_factor
            del pred_data

        logger.info(f"Computing statistics for prediction member {member_idx+1}/{n_members}")
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
            member_result = apply_statistic(member_data, times_dt, stat_config['type'], stat_config['param'], wet_threshold)
            map_output_dir = output_path / f"maps_precip_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'Pred. {member_idx+1}'
            plot_stat_map(member_result, member_filename, stat_config, member_label, grid_cfg, unit=unit)
            if has_target_for_diff:
                target_result = target_results.get(stat_config['stat_name'])
                if target_result is not None:
                    diff = member_result - target_result
                    diff_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_minus_target_{stat_config["stat_name"]}')
                    plot_difference_map(
                        diff,
                        diff_filename,
                        title=f'Pred. {member_idx+1} - Target: {stat_config["title_stat"]} Difference',
                        label=_difference_label(stat_config['type'], unit),
                        grid_cfg=grid_cfg,
                    )

    del member_data
    logger.info("All precipitation statistics maps generated successfully")


if __name__ == '__main__':
    main(parse_eval_cli())