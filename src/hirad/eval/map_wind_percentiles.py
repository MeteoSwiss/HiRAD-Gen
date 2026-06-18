"""Plot per-grid-point percentile maps of 10 m wind speed.
"""
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from hirad.eval.eval_utils import (
    get_channel_indices,
    grid_cfg_from_cfg,
    load_generation_setup,
    parse_eval_cli,
    resolve_ts_dir,
)
from hirad.eval.plotting import plot_difference_map, plot_map


# Percentile maps to compute, with fixed colorbar upper bounds (m/s) so that
# every mode shares the same scale for visual comparison.
PERCENTILE_CONFIG = {
    'p50': {'param': 0.50, 'title': '50th Percentile', 'vmax': 12.0},
    'p90': {'param': 0.90, 'title': '90th Percentile', 'vmax': 18.0},
    'p99': {'param': 0.99, 'title': '99th Percentile', 'vmax': 28.0},
    'p99.9': {'param': 0.999, 'title': '99.9th Percentile', 'vmax': 38.0},
    'p99.99': {'param': 0.9999, 'title': '99.99th Percentile', 'vmax': 48.0},
}


def compute_wind_speed(u, v):
    """Compute wind speed from U and V components."""
    return np.hypot(u, v)


def plot_percentile_map(data, filename, stat_config, label, grid_cfg):
    """Plot a single wind-speed percentile map."""
    plot_map(
        data, filename,
        title=f'{label}: {stat_config["title"]} Wind Speed',
        label='Wind Speed [m/s]',
        vmin=0, vmax=stat_config['vmax'], cmap='inferno', extend='max',
        grid_cfg=grid_cfg,
    )


def compute_fbi_map(mode_speed, target_quantile, frac):
    """Frequency bias index map at one percentile.

    For each grid point this is ``P(pred > q_target(p)) / (1 - p)``: the fraction
    of *mode_speed* timesteps that exceed the target's local percentile threshold,
    normalized by the expected exceedance frequency ``1 - p``. Values near 1 mean
    unbiased exceedance frequency, > 1 over-prediction, < 1 under-prediction.
    """
    exceedance = np.mean(mode_speed > target_quantile[None, :, :], axis=0)
    target_exc = max(1.0 - float(frac), 1e-12)
    return (exceedance / target_exc).astype(np.float32)


def plot_fbi_map(data, filename, stat_config, label, grid_cfg):
    """Plot a single wind-speed frequency-bias-index map (diverging around 1)."""
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0)
    plot_map(
        data, filename,
        title=f'{label}: {stat_config["title"]} Wind Speed FBI',
        label='Frequency Bias Index',
        cmap='RdBu_r', norm=norm, extend='max',
        grid_cfg=grid_cfg,
    )


def _load_speed_stack(times, out_root, mode, u_channel, v_channel, conv_factor,
                      log_interval, logger):
    """Load a ``(T, H, W)`` wind-speed stack for *mode*, computing speed on load."""
    speed_list = []
    for i, ts in enumerate(times):
        if i % log_interval == 0:
            logger.info(f"Loading {mode} timestep {i+1}/{len(times)}: {ts}")
        data = torch.load(resolve_ts_dir(out_root, ts) / ts / f"{ts}-{mode}", weights_only=False)
        u = data[u_channel]
        v = data[v_channel]
        u = (u.numpy() if isinstance(u, torch.Tensor) else u) * conv_factor
        v = (v.numpy() if isinstance(v, torch.Tensor) else v) * conv_factor
        speed_list.append(compute_wind_speed(u, v).astype(np.float32))
        del data, u, v
    return np.stack(speed_list, axis=0)


def main(cfg: dict):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    grid_cfg = grid_cfg_from_cfg(cfg)

    logger.info("Starting wind-speed percentile map generation")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Processing {len(times)} timesteps")

    out_root = Path(generation_dir)
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    indices = get_channel_indices(gen_cfg)

    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    if u10_out is None or v10_out is None:
        logger.error("Wind components (10u / 10v) not found in output channels.")
        return
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)

    conv_factor = cfg.get("wind_conv_factor", 1.0)
    log_interval = cfg.get("log_interval", 100)

    stat_configs = [
        {'stat_name': name, **config}
        for name, config in PERCENTILE_CONFIG.items()
    ]

    # --- Basic modes: target, baseline, regression-prediction ---
    basic_modes = {
        'target': ((u10_out, v10_out), 'Target'),
        'baseline': ((u10_in, v10_in), 'Input'),
        'regression-prediction': ((u10_out, v10_out), 'Regression Prediction'),
    }

    mode_results = {}

    for mode, ((u_channel, v_channel), label) in basic_modes.items():
        logger.info(f"Processing mode: {mode}")
        try:
            mode_speed = _load_speed_stack(
                times, out_root, mode, u_channel, v_channel,
                conv_factor, log_interval, logger,
            )
        except Exception as exc:
            logger.warning(f"{mode} not available, skipping: {exc}")
            continue

        mode_results[mode] = {}
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title']} for {mode}...")
            result = np.quantile(mode_speed, stat_config['param'], axis=0)
            mode_results[mode][stat_config['stat_name']] = result
            map_output_dir = output_path / f"maps_windspeed_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            plot_percentile_map(
                result, str(map_output_dir / f'{mode}_{stat_config["stat_name"]}'),
                stat_config, label, grid_cfg,
            )

        # FBI maps: prediction exceedance vs the target's local percentile threshold.
        # Target is processed first, so its quantiles are available for the others.
        target_quantiles = mode_results.get('target')
        if mode != 'target' and target_quantiles is not None:
            logger.info(f"Computing FBI maps for {mode}...")
            for stat_config in stat_configs:
                stat_name = stat_config['stat_name']
                target_quantile = target_quantiles.get(stat_name)
                if target_quantile is None:
                    continue
                fbi = compute_fbi_map(mode_speed, target_quantile, stat_config['param'])
                map_output_dir = output_path / f"maps_windspeed_{stat_name}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                plot_fbi_map(
                    fbi, str(map_output_dir / f'{mode}_fbi_{stat_name}'),
                    stat_config, label, grid_cfg,
                )

        del mode_speed

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
                map_output_dir = output_path / f"maps_windspeed_{stat_name}"
                map_output_dir.mkdir(parents=True, exist_ok=True)
                plot_difference_map(
                    diff,
                    str(map_output_dir / f'{mode}_minus_target_{stat_name}'),
                    title=f'{label} - Target: {stat_config["title"]} Wind Speed Difference',
                    label='Difference [m/s]',
                    grid_cfg=grid_cfg,
                )

    # --- Predictions: process ONE member at a time to bound memory usage ---
    logger.info("Processing predictions mode...")
    try:
        sample_data = torch.load(resolve_ts_dir(out_root, times[0]) / times[0] / f"{times[0]}-predictions", weights_only=False)
        n_members = sample_data.shape[0]
        del sample_data
    except Exception as exc:
        logger.warning(f"Predictions not available: {exc}")
        logger.info("Wind-speed percentile map generation complete")
        return
    logger.info(f"Found {n_members} ensemble members")

    H: int = cfg["height"]
    W: int = cfg["width"]
    member_speed = np.empty((len(times), H, W), dtype=np.float32)
    has_target_for_diff = target_results is not None
    if not has_target_for_diff:
        logger.warning("Target mode not available; skipping prediction-minus-target difference maps for members")

    for member_idx in range(n_members):
        logger.info(f"Loading prediction member {member_idx+1}/{n_members} (single pass over files)...")
        for i, ts in enumerate(times):
            if i % log_interval == 0:
                logger.info(f"Loading predictions member {member_idx+1} timestep {i+1}/{len(times)}: {ts}")
            pred_data = torch.load(resolve_ts_dir(out_root, ts) / ts / f"{ts}-predictions", weights_only=False)
            u = pred_data[member_idx, u10_out]
            v = pred_data[member_idx, v10_out]
            u = (u.numpy() if isinstance(u, torch.Tensor) else u) * conv_factor
            v = (v.numpy() if isinstance(v, torch.Tensor) else v) * conv_factor
            member_speed[i] = compute_wind_speed(u, v)
            del pred_data, u, v

        logger.info(f"Computing percentiles for prediction member {member_idx+1}/{n_members}")
        for stat_config in stat_configs:
            logger.info(f"Computing {stat_config['title']} for member {member_idx+1}...")
            member_result = np.quantile(member_speed, stat_config['param'], axis=0)
            map_output_dir = output_path / f"maps_windspeed_{stat_config['stat_name']}"
            map_output_dir.mkdir(parents=True, exist_ok=True)
            member_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_{stat_config["stat_name"]}')
            member_label = f'CorrDiff Member {member_idx+1}'
            plot_percentile_map(member_result, member_filename, stat_config, member_label, grid_cfg)
            if has_target_for_diff:
                target_result = target_results.get(stat_config['stat_name'])
                if target_result is not None:
                    diff = member_result - target_result
                    diff_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_minus_target_{stat_config["stat_name"]}')
                    plot_difference_map(
                        diff,
                        diff_filename,
                        title=f'CorrDiff Member {member_idx+1} - Target: {stat_config["title"]} Wind Speed Difference',
                        label='Difference [m/s]',
                        grid_cfg=grid_cfg,
                    )
                    fbi = compute_fbi_map(member_speed, target_result, stat_config['param'])
                    fbi_filename = str(map_output_dir / f'prediction_member_{member_idx:02d}_fbi_{stat_config["stat_name"]}')
                    plot_fbi_map(fbi, fbi_filename, stat_config, member_label, grid_cfg)

    del member_speed
    logger.info("Wind-speed percentile map generation complete")


if __name__ == '__main__':
    main(parse_eval_cli())
