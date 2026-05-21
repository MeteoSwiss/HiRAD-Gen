"""Probability of exceedance for wind speed and components."""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.eval_utils import load_generation_setup, parse_eval_cli, resolve_ts_dir
from hirad.eval.plotting import get_channel_indices
from hirad.eval.eval_utils import percentiles_from_histogram


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
    """Update exceedance counts incrementally."""
    data = np.abs(values) if use_abs else values
    counts += (data[:, None] > thresholds[None, :]).sum(axis=0)
    total += len(values)
    return counts, total


def compute_percentiles(values, percentile_dict, use_abs=False):
    """Compute percentiles."""
    data = np.abs(values) if use_abs else values
    data_array = xr.DataArray(data)
    return {key: data_array.quantile(p).item() for key, p in percentile_dict.items()}


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

    # Initialize dataset
    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.get("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    # Output root
    out_root = Path(generation_dir)

    # Find channel indices for wind components
    indices = get_channel_indices(dataset)
    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)
    
    if u10_out is None or v10_out is None:
        logger.error("Wind components (10u, 10v) not found in dataset!")
        return
    
    logger.info(f"Wind component channel indices - output: 10u={u10_out}, 10v={v10_out}, input: 10u={u10_in}, 10v={v10_in}")

    # Define thresholds for exceedance calculation (same for all variables)
    thresholds = np.logspace(-1, 2, 200)  # From 0.1 to ~100 m/s
    n_thresholds = len(thresholds)

    # Histogram bins for percentile estimation (fine-grained log-spaced)
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-1, 2.5, 5000) # From 0.1 to ~316 m/s
    ])
    n_hist_bins = len(hist_bins) - 1
    
    # Storage for exceedance counts (incremental computation)
    exceedance_counts = {
        'speed': {}, 'u': {}, 'v': {}
    }
    totals = {'speed': {}, 'u': {}, 'v': {}}
    hist_counts = {
        'speed': {}, 'u': {}, 'v': {}
    }
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        # Initialize counts
        for var in ['speed', 'u', 'v']:
            exceedance_counts[var][mode] = np.zeros(n_thresholds, dtype=np.int64)
            totals[var][mode] = 0
            hist_counts[var][mode] = np.zeros(n_hist_bins, dtype=np.int64)
        
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"Processing timestep {i+1}/{len(times)}")
                
                data = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-{mode}", weights_only=False)
                
                # Extract wind components
                if mode in ['target', 'regression-prediction']:
                    u = data[u10_out]
                    v = data[v10_out]
                else:  # baseline
                    u = data[u10_in]
                    v = data[v10_in]
                
                wind_speed = compute_wind_speed(u, v)
                
                # Get valid values
                valid_mask = ~np.isnan(wind_speed)
                speed_vals = wind_speed[valid_mask].flatten()
                u_vals = u[valid_mask].flatten()
                v_vals = v[valid_mask].flatten()
                
                # Update exceedance counts incrementally
                exceedance_counts['speed'][mode], totals['speed'][mode] = update_exceedance_counts(
                    exceedance_counts['speed'][mode], totals['speed'][mode], speed_vals, thresholds, use_abs=False
                )
                exceedance_counts['u'][mode], totals['u'][mode] = update_exceedance_counts(
                    exceedance_counts['u'][mode], totals['u'][mode], u_vals, thresholds, use_abs=True
                )
                exceedance_counts['v'][mode], totals['v'][mode] = update_exceedance_counts(
                    exceedance_counts['v'][mode], totals['v'][mode], v_vals, thresholds, use_abs=True
                )
                
                # Collect samples for percentiles (subsample to save memory)
                hist_counts['speed'][mode] += np.histogram(speed_vals, bins=hist_bins)[0]
                hist_counts['u'][mode] += np.histogram(np.abs(u_vals), bins=hist_bins)[0]
                hist_counts['v'][mode] += np.histogram(np.abs(v_vals), bins=hist_bins)[0]
                
        except Exception as e:
            logger.warning(f"{mode} data not found or error occurred, skipping: {e}")
            continue

        logger.info(f"Processed {totals['speed'][mode]} values for {mode}")
            
    # -- Process predictions: compute exceedance for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    member_counts = {'speed': [], 'u': [], 'v': []}
    member_totals = {'speed': [], 'u': [], 'v': []}
    member_hist_counts = {'speed': [], 'u': [], 'v': []}
    
    for i, ts in enumerate(times):
        if i % cfg.get("log_interval") == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False)  # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            for var in ['speed', 'u', 'v']:
                member_counts[var] = [np.zeros(n_thresholds, dtype=np.int64) for _ in range(n_members)]
                member_totals[var] = [0 for _ in range(n_members)]
                member_hist_counts[var] = [np.zeros(n_hist_bins, dtype=np.int64) for _ in range(n_members)]
        
        for member_idx in range(n_members):
            u = preds[member_idx, u10_out]
            v = preds[member_idx, v10_out]
            wind_speed = compute_wind_speed(u, v)
            
            valid_mask = ~np.isnan(wind_speed)
            speed_vals = wind_speed[valid_mask].flatten()
            u_vals = u[valid_mask].flatten()
            v_vals = v[valid_mask].flatten()
            
            # Update counts
            member_counts['speed'][member_idx], member_totals['speed'][member_idx] = update_exceedance_counts(
                member_counts['speed'][member_idx], member_totals['speed'][member_idx], speed_vals, thresholds, use_abs=False
            )
            member_counts['u'][member_idx], member_totals['u'][member_idx] = update_exceedance_counts(
                member_counts['u'][member_idx], member_totals['u'][member_idx], u_vals, thresholds, use_abs=True
            )
            member_counts['v'][member_idx], member_totals['v'][member_idx] = update_exceedance_counts(
                member_counts['v'][member_idx], member_totals['v'][member_idx], v_vals, thresholds, use_abs=True
            )
            
            # Collect samples for percentiles
            member_hist_counts['speed'][member_idx] += np.histogram(speed_vals, bins=hist_bins)[0]
            member_hist_counts['u'][member_idx] += np.histogram(np.abs(u_vals), bins=hist_bins)[0]
            member_hist_counts['v'][member_idx] += np.histogram(np.abs(v_vals), bins=hist_bins)[0]
    
    logger.info(f"Collected {n_members} ensemble members for predictions")
    
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
