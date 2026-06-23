"""Probability of exceedance for wind speed and components."""
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hirad.eval.eval_utils import get_channel_indices, load_generation_setup, parse_eval_cli, resolve_ts_dir
from hirad.eval.eval_utils import percentiles_from_histogram, FONT_SIZE

# Presentation-sized fonts for all figures in this script.
plt.rcParams.update(FONT_SIZE)


def compute_wind_speed(u, v):
    """Compute wind speed from U and V components."""
    return np.hypot(u, v)


def compute_exceedance_probs(values, thresholds, use_abs=False):
    """Compute exceedance probabilities."""
    if use_abs:
        return np.array([np.mean(np.abs(values) > t) for t in thresholds])
    else:
        return np.array([np.mean(values > t) for t in thresholds])


def update_exceedance_counts(counts, total, values, thresholds, use_abs=False):
    """Update exceedance counts incrementally via searchsorted (O(n log k))."""
    data = np.abs(values) if use_abs else values
    # idx[i] = number of thresholds strictly less than data[i] (side='left')
    # data[i] > thresholds[j]  iff  idx[i] > j
    idx = np.searchsorted(thresholds, data, side='left')
    bin_counts = np.bincount(idx, minlength=len(thresholds) + 1)
    counts += len(data) - np.cumsum(bin_counts)[: len(thresholds)]
    total += len(data)
    return counts, total


def compute_percentiles(values, percentile_dict, use_abs=False):
    """Compute percentiles."""
    data = np.abs(values) if use_abs else values
    return {key: float(np.quantile(data, p)) for key, p in percentile_dict.items()}


def save_exceedance_plot(exceedance_data_dict, thresholds, labels, colors, title, ylabel, out_path, percentiles_data=None):
    """Save probability of exceedance plot."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot exceedance curves
    for (key, exceedance_data), label, color in zip(exceedance_data_dict.items(), labels, colors):
        if isinstance(exceedance_data, tuple):  # Handle ensemble data
            # Plot individual members with transparency
            for i, member_exceedance in enumerate(exceedance_data):
                alpha = 0.5 if i > 0 else 0.7
                label_member = label if i == 0 else None
                plt.plot(thresholds, member_exceedance, alpha=alpha, color=color, 
                        label=label_member, linewidth=1)
        else:
            # Plot single dataset
            plt.plot(thresholds, exceedance_data, alpha=0.7, color=color, 
                    label=label, linewidth=2)
    
    plt.xscale('log')
    plt.xlim(thresholds[1], thresholds[-1])
    plt.yscale('log')
    plt.xlabel(ylabel)
    plt.ylabel('Probability of Exceedance')
    plt.ylim(1e-8, 1)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add percentile lines if provided
    if percentiles_data:        
        # Calculate y-range for percentile lines (lowest 10% of log scale)
        y_bottom, y_top = plt.ylim()
        log_bottom, log_top = np.log10(y_bottom), np.log10(y_top)
        vline_ymax = 10**(log_bottom + 0.1 * (log_top - log_bottom))
        vline_ymin = y_bottom
        
        # Define line styles for percentiles
        percentile_styles = {99: '--', 99.9: ':', 99.99: '-.'}
        percentile_labels = {99: '99th all-hour percentiles', 99.9: '99.9th all-hour percentiles', 99.99: '99.99th all-hour percentiles'}
        colors_perc = {'target': 'blue', 'baseline': 'orange', 'predictions': 'green', 'regression-prediction': 'red'}
        legend_added = set()
        
        # Plot all percentile lines
        for dataset_name, data in percentiles_data.items():
            color = colors_perc[dataset_name]
            
            if dataset_name in ['target', 'baseline', 'regression-prediction']:
                # Single dataset
                for percentile, value in data.items():
                    linestyle = percentile_styles[percentile]
                    legend_added.add(percentile)  # Track percentiles for black legend entries
                    
                    plt.vlines(x=value, colors=color, ymin=vline_ymin, ymax=vline_ymax,
                              linestyles=linestyle, alpha=0.8)  # No label here
            else:
                # Ensemble members
                for member_data in data.values():
                    for percentile, value in member_data.items():
                        linestyle = percentile_styles[percentile]
                        legend_added.add(percentile)  # Track percentiles for black legend entries
                        
                        plt.vlines(x=value, colors=color, ymin=vline_ymin, ymax=vline_ymax,
                                  linestyles=linestyle, alpha=0.6)  # No label here
        
        # Add black legend entries for percentiles (override the colored ones)
        for percentile in [99, 99.9, 99.99]:
            if percentile in legend_added:
                plt.plot([], [], color='black', linestyle=percentile_styles[percentile], 
                        label=percentile_labels[percentile])
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def _accumulate(exc_counts, h_counts, speed_vals, u_vals, v_vals,
                thresholds, hist_interior, n_hist_bins):
    """Update exceedance and histogram count arrays in-place."""
    abs_u = np.abs(u_vals)
    abs_v = np.abs(v_vals)
    n_thr = len(thresholds)

    # Exceedance counts: O(n log k) via searchsorted+bincount
    for counts, data in [
        (exc_counts['speed'], speed_vals),
        (exc_counts['u'], abs_u),
        (exc_counts['v'], abs_v),
    ]:
        idx = np.searchsorted(thresholds, data, side='left')
        bin_counts = np.bincount(idx, minlength=n_thr + 1)
        counts += len(data) - np.cumsum(bin_counts)[:n_thr]

    # Histogram counts: searchsorted+bincount, no temporary bool array
    for hcounts, data in [
        (h_counts['speed'], speed_vals),
        (h_counts['u'], abs_u),
        (h_counts['v'], abs_v),
    ]:
        hcounts += np.bincount(
            np.searchsorted(hist_interior, data, side='right'), minlength=n_hist_bins
        )


def main(cfg: dict):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for probability of exceedance for wind speed")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Output root
    out_root = Path(generation_dir)

    # Find channel indices for wind components
    indices = get_channel_indices(gen_cfg)
    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)

    if u10_out is None or v10_out is None:
        logger.error("Wind components (10u, 10v) not found in dataset!")
        return

    logger.info(f"Wind component channel indices - output: 10u={u10_out}, 10v={v10_out}, "
                f"input: 10u={u10_in}, 10v={v10_in}")

    # Define thresholds for exceedance calculation (same for all variables)
    thresholds = np.logspace(-1, 2, 200)  # From 0.1 to ~100 m/s
    n_thresholds = len(thresholds)

    # Histogram bins for percentile estimation (fine-grained log-spaced)
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-1, 2.5, 5000)  # From 0.1 to ~316 m/s
    ])
    n_hist_bins = len(hist_bins) - 1
    # Interior edges used by searchsorted+bincount (equivalent to np.histogram)
    hist_interior = hist_bins[1:-1]

    det_modes = ('target', 'baseline', 'regression-prediction')
    VARS = ('speed', 'u', 'v')

    # Storage for exceedance counts and histograms
    exceedance_counts = {var: {} for var in VARS}
    totals            = {var: {} for var in VARS}
    hist_counts       = {var: {} for var in VARS}
    for mode in det_modes:
        for var in VARS:
            exceedance_counts[var][mode] = np.zeros(n_thresholds, dtype=np.int64)
            totals[var][mode]            = 0
            hist_counts[var][mode]       = np.zeros(n_hist_bins,  dtype=np.int64)

    n_members         = None
    member_counts     = {var: [] for var in VARS}
    member_totals     = {var: [] for var in VARS}
    member_hist_counts = {var: [] for var in VARS}

    skip_det  = set()   # modes whose files were missing
    skip_preds = False

    log_interval = cfg.get("log_interval", 24)

    # Pre-resolve timestamp directories once (avoids 4× filesystem glob per timestep)
    logger.info(f"Resolving {len(times)} timestep directories ...")
    ts_dirs = {ts: resolve_ts_dir(out_root, ts) / ts for ts in times}

    # -- Single pass over all timesteps --
    t0 = time.perf_counter()
    for i, ts in enumerate(times):
        if i % log_interval == 0:
            elapsed = time.perf_counter() - t0
            logger.info(f"Processing timestep {i+1}/{len(times)}  ({elapsed:.1f}s elapsed)")

        ts_dir = ts_dirs[ts]

        # Deterministic modes: target, baseline, regression-prediction
        for mode in det_modes:
            if mode in skip_det:
                continue
            try:
                data = torch.load(ts_dir / f"{ts}-{mode}", weights_only=False)
            except FileNotFoundError:
                logger.warning(f"[{mode}] file not found at {ts}, skipping mode entirely")
                skip_det.add(mode)
                continue

            ch_u = u10_in if mode == 'baseline' else u10_out
            ch_v = v10_in if mode == 'baseline' else v10_out
            u = data[ch_u]
            v = data[ch_v]

            wind_speed = compute_wind_speed(u, v)
            valid_mask = ~np.isnan(wind_speed)
            speed_vals = wind_speed[valid_mask].ravel()
            u_vals     = u[valid_mask].ravel()
            v_vals     = v[valid_mask].ravel()

            _accumulate(
                {var: exceedance_counts[var][mode] for var in VARS},
                {var: hist_counts[var][mode] for var in VARS},
                speed_vals, u_vals, v_vals,
                thresholds, hist_interior, n_hist_bins,
            )
            totals['speed'][mode] += len(speed_vals)
            totals['u'][mode]     += len(u_vals)
            totals['v'][mode]     += len(v_vals)

        # Predictions (ensemble)
        if not skip_preds:
            try:
                preds = torch.load(ts_dir / f"{ts}-predictions", weights_only=False)  # [M, C, H, W]
            except FileNotFoundError:
                logger.warning(f"[predictions] file not found at {ts}, skipping ensemble entirely")
                skip_preds = True
                continue

            if n_members is None:
                n_members = preds.shape[0]
                logger.info(f"Detected {n_members} ensemble members")
                for var in VARS:
                    member_counts[var]      = [np.zeros(n_thresholds, dtype=np.int64) for _ in range(n_members)]
                    member_totals[var]      = [0] * n_members
                    member_hist_counts[var] = [np.zeros(n_hist_bins,  dtype=np.int64) for _ in range(n_members)]

            for m in range(n_members):
                u = preds[m, u10_out]
                v = preds[m, v10_out]
                wind_speed = compute_wind_speed(u, v)
                valid_mask = ~np.isnan(wind_speed)
                speed_vals = wind_speed[valid_mask].ravel()
                u_vals     = u[valid_mask].ravel()
                v_vals     = v[valid_mask].ravel()

                _accumulate(
                    {var: member_counts[var][m] for var in VARS},
                    {var: member_hist_counts[var][m] for var in VARS},
                    speed_vals, u_vals, v_vals,
                    thresholds, hist_interior, n_hist_bins,
                )
                member_totals['speed'][m] += len(speed_vals)
                member_totals['u'][m]     += len(u_vals)
                member_totals['v'][m]     += len(v_vals)

    total_elapsed = time.perf_counter() - t0
    logger.info(f"Single-pass loop completed in {total_elapsed:.1f}s for {len(times)} timesteps")
    if n_members is not None:
        logger.info(f"Collected {n_members} ensemble members for predictions")
    else:
        n_members = 0  # no predictions found; guard range() calls below

    # Convert counts to probabilities
    exceedance_data = {'speed': {}, 'u': {}, 'v': {}}
    
    for var in ['speed', 'u', 'v']:
        # Single datasets
        for mode in ['target', 'baseline', 'regression-prediction']:
            if mode in exceedance_counts[var] and totals[var][mode] > 0:
                exceedance_data[var][mode] = exceedance_counts[var][mode] / totals[var][mode]
        
        # Ensemble members
        member_probs = []
        for member_idx in range(n_members):
            if member_totals[var][member_idx] > 0:
                member_probs.append(member_counts[var][member_idx] / member_totals[var][member_idx])
        exceedance_data[var]['predictions'] = tuple(member_probs)
    
    # Compute percentiles for all datasets and variables
    percentiles = {99: 0.99, 99.9: 0.999, 99.99: 0.9999}
    percentiles_data = {'speed': {}, 'u': {}, 'v': {}}
    
    # Single datasets (target, baseline, regression-prediction)
    for var in ['speed', 'u', 'v']:
        for mode in ['target', 'baseline', 'regression-prediction']:
            if mode in hist_counts[var] and totals[var][mode] > 0:
                percentiles_data[var][mode] = percentiles_from_histogram(
                    hist_counts[var][mode], hist_bins, percentiles
                )
        
        # Ensemble members
        percentiles_data[var]['predictions'] = {}
        for member_idx in range(n_members):
            if member_totals[var][member_idx] > 0:
                percentiles_data[var]['predictions'][f'member_{member_idx}'] = percentiles_from_histogram(
                    member_hist_counts[var][member_idx], hist_bins, percentiles
                )
    
    # Create exceedance plots
    labels = ['Target', 'Input', 'Regression Prediction', 'CorrDiff Ensemble'] if 'regression-prediction' in exceedance_data['speed'] else ['Target', 'Input', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'red', 'green'] if 'regression-prediction' in exceedance_data['speed'] else ['blue', 'orange', 'green']
    
    # Define plot configurations
    plot_configs = [
        ('windspeed_exceedance.png', 'speed', 'Probability of Exceedance for Wind Speed', 
         'All-hour Wind Speed [m/s] (Pooled Data)'),
        ('wind_u_exceedance.png', 'u', 'Probability of Exceedance for abs(10u)', 
         'All-hour 10u Component [m/s] (Pooled Data)'),
        ('wind_v_exceedance.png', 'v', 'Probability of Exceedance for abs(10v)', 
         'All-hour 10v Component [m/s] (Pooled Data)'),
    ]
    
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, var, title, ylabel in plot_configs:
        fn = output_path / filename
        save_exceedance_plot(
            exceedance_data[var],
            thresholds,
            labels,
            colors,
            title,
            ylabel,
            fn,
            percentiles_data[var]
        )
        logger.info(f"{var.capitalize()} exceedance plot saved: {fn}")


if __name__ == '__main__':
    main(parse_eval_cli())
