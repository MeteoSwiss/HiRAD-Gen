import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import numba

from hirad.eval.eval_utils import get_channel_indices, grid_cfg_from_cfg, load_generation_setup, parse_eval_cli, resolve_ts_dir
from hirad.eval.plotting import plot_map, plot_map_temperature


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


def apply_statistic(data_np, times_dt, stat_type, stat_param=None):
    """
    Apply temperature statistic on array containing time sequence of 2m temperature maps.
    data_np: (T, H, W) float array in degrees Celsius
    times_dt: list of datetime objects (length T)
    Returns: (H, W) numpy array
    """
    if stat_type == 'mean':
        return np.mean(data_np, axis=0)

    if stat_type == 'std':
        return np.std(data_np, axis=0)

    if stat_type == 'max':
        # TXx: maximum temperature
        return np.max(data_np, axis=0)

    if stat_type == 'min':
        # TNn: minimum temperature
        return np.min(data_np, axis=0)

    if stat_type == 'quantile':
        return np.quantile(data_np, stat_param, axis=0)

    # For daily-based indices, build daily aggregations using xarray
    if stat_type in ('warm_days', 'frost_days', 'ice_days', 'tropical_nights',
                     'dtr', 'warm_spell', 'cold_spell'):
        da = xr.DataArray(
            data_np, dims=['time', 'lat', 'lon'],
            coords={'time': times_dt}
        )
        daily_max = da.resample(time="1D").max("time").values   # (D, H, W)
        daily_min = da.resample(time="1D").min("time").values   # (D, H, W)
        D = daily_max.shape[0]

        if stat_type == 'warm_days':
            # SU: fraction of days with daily max > 25 °C
            return np.mean(daily_max > 25.0, axis=0) * 100.0

        if stat_type == 'frost_days':
            # FD: fraction of days with daily min < 0 °C
            return np.mean(daily_min < 0.0, axis=0) * 100.0

        if stat_type == 'ice_days':
            # ID: fraction of days with daily max < 0 °C
            return np.mean(daily_max < 0.0, axis=0) * 100.0

        if stat_type == 'tropical_nights':
            # TR: fraction of days with daily min > 20 °C
            return np.mean(daily_min > 20.0, axis=0) * 100.0

        if stat_type == 'dtr':
            # DTR: mean diurnal temperature range
            return np.mean(daily_max - daily_min, axis=0)

        if stat_type == 'warm_spell':
            # WSDI-like: longest consecutive run of days with daily max > 25 °C
            return consecutive_spell(daily_max, lambda x: x > 25.0)

        if stat_type == 'cold_spell':
            # CSDI-like: longest consecutive run of days with daily min < 0 °C
            return consecutive_spell(daily_min, lambda x: x < 0.0)

    raise ValueError(f"Unsupported temperature statistic type: {stat_type}")


def plot_temp_stat_map(data, filename, stat_config, label, grid_cfg):
    """Plot a single temperature statistic map with appropriate styling."""
    stype = stat_config['type']
    title = f'{label}: {stat_config["title_stat"]}'

    if stype in ('mean', 'quantile', 'max', 'min'):
        plot_map_temperature(data, filename, title=title, grid_cfg=grid_cfg)
    elif stype == 'std':
        plot_map(
            data, filename,
            title=title,
            label='Std Dev [°C]',
            vmin=0, vmax=10, cmap='plasma', extend='max', grid_cfg=grid_cfg
        )
    elif stype == 'dtr':
        plot_map(
            data, filename,
            title=title,
            label='Diurnal Range [°C]',
            vmin=0, vmax=20, cmap='YlOrRd', extend='max', grid_cfg=grid_cfg
        )
    elif stype in ('warm_days', 'frost_days', 'ice_days', 'tropical_nights'):
        plot_map(
            data, filename,
            title=title,
            label='Frequency [% of days]',
            vmin=0, vmax=100, cmap='OrRd', extend='neither', grid_cfg=grid_cfg
        )
    elif stype == 'warm_spell':
        plot_map(
            data, filename,
            title=title,
            label='Days',
            vmin=0, vmax=60, cmap='YlOrRd', extend='max', grid_cfg=grid_cfg
        )
    elif stype == 'cold_spell':
        plot_map(
            data, filename,
            title=title,
            label='Days',
            vmin=0, vmax=30, cmap='YlGnBu', extend='max', grid_cfg=grid_cfg
        )
    else:
        plot_map(
            data, filename,
            title=title,
            label='Temperature [°C]',
            vmin=None, vmax=None, cmap='RdBu_r', extend='both', grid_cfg=grid_cfg
        )


def main(cfg: dict):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    grid_cfg = grid_cfg_from_cfg(cfg)

    logger.info("Starting 2m temperature statistics generation")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Processing {len(times)} timesteps")

    times_dt = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)

    indices = get_channel_indices(gen_cfg)
    out_ch = indices['output']
    in_ch = indices['input']

    # Temperature channel: try '2t', fall back to 't2m'
    t2m_out = out_ch.get('2t', out_ch.get('t2m'))
    t2m_in = in_ch.get('2t', in_ch.get('t2m', t2m_out))

    if t2m_out is None:
        logger.error("No temperature channel ('2t' or 't2m') found in output channels. Aborting.")
        return

    # Conversion from Kelvin to Celsius: value * temp_conv_factor + temp_conv_offset
    conv_factor = cfg.get("temp_conv_factor", 1.0)
    conv_offset = cfg.get("temp_conv_offset", -273.15)
    log_interval = cfg.get("log_interval", 100)

    STATISTICS_CONFIG = {
        'mean':            {'type': 'mean',            'title': 'Mean Temperature'},
        'std':             {'type': 'std',             'title': 'Temperature Variability (Std Dev)'},
        'txx':             {'type': 'max',             'title': 'Maximum Temperature (TXx)'},
        'tnn':             {'type': 'min',             'title': 'Minimum Temperature (TNn)'},
        'p99.99':          {'type': 'quantile', 'param': 0.9999, 'title': '99.99th Percentile Temperature'},
        'p99.9':           {'type': 'quantile', 'param': 0.999, 'title': '99.9th Percentile Temperature'},
        'p99':             {'type': 'quantile', 'param': 0.99,  'title': '99th Percentile Temperature'},
        'p01':             {'type': 'quantile', 'param': 0.01,  'title': '1st Percentile Temperature'},
        'p0.1':            {'type': 'quantile', 'param': 0.001,  'title': '0.1th Percentile Temperature'},
        'p0.01':           {'type': 'quantile', 'param': 0.0001, 'title': '0.01th Percentile Temperature'},
        'warm_days':       {'type': 'warm_days',       'title': 'Summer Days (daily max > 25°C)'},
        'frost_days':      {'type': 'frost_days',      'title': 'Frost Days (daily min < 0°C)'},
        'ice_days':        {'type': 'ice_days',        'title': 'Ice Days (daily max < 0°C)'},
        'tropical_nights': {'type': 'tropical_nights', 'title': 'Tropical Nights (daily min > 20°C)'},
        'dtr':             {'type': 'dtr',             'title': 'Mean Diurnal Temperature Range (DTR)'},
        'warm_spell':      {'type': 'warm_spell',      'title': 'Warm Spell Duration (daily max > 25°C)'},
        'cold_spell':      {'type': 'cold_spell',      'title': 'Cold Spell Duration (daily min < 0°C)'},
    }
    stat_configs = [
        {'stat_name': name, 'title_stat': config['title'], 'param': config.get('param'), **config}
        for name, config in STATISTICS_CONFIG.items()
    ]

    # --- Basic modes: target, baseline, regression-prediction ---
    basic_modes = {
        'target':                (t2m_out, 'Target'),
        'baseline':              (t2m_in,  'Input'),
        'regression-prediction': (t2m_out, 'Regression Prediction'),
    }

    for mode, (t2m_channel, label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        data_list = []
        try:
            for i, ts in enumerate(times):
                if i % log_interval == 0:
                    logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
                raw = torch.load(resolve_ts_dir(out_root, ts) / ts / f"{ts}-{mode}", weights_only=False)
                val = raw[t2m_channel]
                arr = val.numpy() if isinstance(val, torch.Tensor) else val
                data_list.append(arr * conv_factor + conv_offset)
        except Exception:
            logger.warning(f"{mode} not available, skipping")
            continue

        mode_data = np.stack(data_list, axis=0).astype(np.float32)
        del data_list

        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for {mode}...")
            result = apply_statistic(mode_data, times_dt, stat_config['type'], stat_config['param'])
            map_output_dir = output_path / f"maps_temp_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_temp_stat_map(result, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'), stat_config, label, grid_cfg)

        del mode_data

    # --- Predictions: process ONE member at a time to bound memory usage ---
    logger.info("Processing predictions mode...")
    try:
        sample_data = torch.load(
            resolve_ts_dir(out_root, times[0]) / times[0] / f"{times[0]}-predictions",
            weights_only=False
        )
        n_members = sample_data.shape[0]
        del sample_data
    except Exception as exc:
        logger.error(f"Could not load predictions: {exc}")
        return
    logger.info(f"Found {n_members} ensemble members")

    H: int = cfg["height"]
    W: int = cfg["width"]
    member_data = np.empty((len(times), H, W), dtype=np.float32)

    for member_idx in range(n_members):
        logger.info(f"Loading prediction member {member_idx+1}/{n_members} (single pass over files)...")
        for i, ts in enumerate(times):
            if i % log_interval == 0:
                logger.info(f"Loading predictions member {member_idx+1} timestep {i+1}/{len(times)}: {ts}")
            pred_data = torch.load(
                resolve_ts_dir(out_root, ts) / ts / f"{ts}-predictions",
                weights_only=False
            )
            member_slice = pred_data[member_idx, t2m_out]
            arr = member_slice.numpy() if isinstance(member_slice, torch.Tensor) else member_slice
            member_data[i] = arr * conv_factor + conv_offset
            del pred_data

        logger.info(f"Computing statistics for prediction member {member_idx+1}/{n_members}")
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title_stat']} for member {member_idx+1}...")
            member_result = apply_statistic(member_data, times_dt, stat_config['type'], stat_config['param'])
            map_output_dir = output_path / f"maps_temp_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'CorrDiff Member {member_idx+1}'
            plot_temp_stat_map(member_result, member_filename, stat_config, member_label, grid_cfg)

    del member_data
    logger.info("All 2m temperature statistics maps generated successfully")


if __name__ == '__main__':
    main(parse_eval_cli())
