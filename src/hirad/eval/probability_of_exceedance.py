"""
Plots the probability of exceedance for precipitation over land.

This script computes and visualizes the complementary cumulative distribution
(probability of exceeding x mm/h) over land).
"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.datasets import get_channels_from_strings, get_strings_from_channels
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.eval_utils import get_channel_indices, load_generation_setup, load_land_sea_mask, relax_zone_interior_mask, parse_eval_cli, precip_conv_factor, resolve_ts_dir
from hirad.eval.eval_utils import percentiles_from_histogram, FONT_SIZE

# Presentation-sized fonts for all figures in this script.
plt.rcParams.update(FONT_SIZE)


def save_exceedance_plot(exceedance_data_dict, thresholds, labels, colors, title, ylabel, out_path, percentiles_data=None):
    """Save probability of exceedance plot with pre-computed data."""
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
    plt.yscale('log')
    plt.xlabel(ylabel)
    plt.ylabel('Probability of Exceedance')
    plt.ylim(1e-8, 1)
    plt.xlim(thresholds[1], thresholds[-1])
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

    logger.info("Starting computation for probability of exceedance over land")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Output root
    out_root = Path(generation_dir)

    # Find channel indices
    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask (sea = NaN); the relaxation zone is dropped separately.
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))
    land_mask = land_mask.where(relax_zone_interior_mask(cfg.get("height"), cfg.get("width"), cfg.get("relax_zone")))

    conv_factor = precip_conv_factor(cfg)  # mm/h

    # Define thresholds for exceedance calculation
    thresholds = np.logspace(-2, 3.0, 200)  # From 0.01 to 1000 mm/h
    n_thresholds = len(thresholds)

    # Histogram bins for percentile estimation (fine-grained log-spaced)
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-2, 3.2, 5000) # From 0.01 to ~1585 mm/h
    ])
    n_hist_bins = len(hist_bins) - 1

    # Storage for exceedance data and land values
    exceedance_counts = {}
    totals = {}
    hist_counts = {}  # For percentile estimation
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        mode_exc_counts = np.zeros(n_thresholds, dtype=np.int64)
        mode_total = 0
        mode_hist = np.zeros(n_hist_bins, dtype=np.int64)
        
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"Processing timestep {i+1}/{len(times)}")
                
                data = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-{mode}", weights_only=False)[tp_out if mode in ['target','regression-prediction'] else tp_in] * conv_factor * land_mask
                
                land_values = data.values[~np.isnan(data.values)]
                n_vals = len(land_values)
                mode_total += n_vals
                # Update counts for exceedance calculation
                mode_exc_counts += np.sum(
                    land_values[:, None] > thresholds[None, :], axis=0
                )
                # Update histogram counts for percentile estimation
                mode_hist += np.histogram(land_values, bins=hist_bins)[0]
        except:
            logger.warning(f"{mode} data not found, skipping")
            continue

        # Compute exceedance probabilities
        exceedance_counts[mode] = mode_exc_counts
        totals[mode] = mode_total
        hist_counts[mode] = mode_hist
        logger.info(f"Processed {mode_total} land values for {mode}")
            
    # -- Process predictions: compute exceedance for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    member_exc_counts = None
    member_totals = None
    member_hist = None
    
    for i, ts in enumerate(times):
        if i % cfg.get("log_interval") == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False) * conv_factor # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            member_exc_counts = [np.zeros(n_thresholds, dtype=np.int64) for _ in range(n_members)]
            member_totals = [0] * n_members
            member_hist = [np.zeros(n_hist_bins, dtype=np.int64) for _ in range(n_members)]
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
            land_values = data.values[~np.isnan(data.values)]
            n_vals = len(land_values)
            member_totals[member_idx] += n_vals
            member_exc_counts[member_idx] += np.sum(
                land_values[:, None] > thresholds[None, :], axis=0
            )
            member_hist[member_idx] += np.histogram(land_values, bins=hist_bins)[0]
    
    # Compute exceedance probabilities
    exceedance_data = {}
    for mode in ['target', 'baseline', 'regression-prediction']:
        if mode in exceedance_counts and totals[mode] > 0:
            exceedance_data[mode] = exceedance_counts[mode] / totals[mode]

    member_exceedance_data = []
    for member_idx in range(n_members):
        if member_totals[member_idx] > 0:
            member_exceedance_data.append(
                member_exc_counts[member_idx] / member_totals[member_idx]
            )
    
    exceedance_data['predictions'] = tuple(member_exceedance_data)
    
    logger.info(f"Collected {n_members} ensemble members for predictions")
    
    # Compute percentiles for all datasets
    percentiles_data = {}
    percentiles = {99: 0.99, 99.9: 0.999, 99.99: 0.9999}

    # Estimating percentiles from fine-grained histograms
    for mode in ['target', 'baseline', 'regression-prediction']:
        if mode in hist_counts and totals[mode] > 0:
            percentiles_data[mode] = percentiles_from_histogram(
                hist_counts[mode], hist_bins, percentiles
            )

    percentiles_data['predictions'] = {}
    for member_idx in range(n_members):
        if member_totals[member_idx] > 0:
            percentiles_data['predictions'][f'member_{member_idx}'] = percentiles_from_histogram(
                member_hist[member_idx], hist_bins, percentiles
            )
    
    # Create exceedance plots
    labels = ['Target', 'Input', 'Regression Prediction', 'CorrDiff Ensemble'] if 'regression-prediction' in exceedance_data else ['Target', 'Input', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'red', 'green'] if 'regression-prediction' in exceedance_data else ['blue', 'orange', 'green']
    
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    fn = output_path / 'precipitation_exceedance_over_land.png'
    save_exceedance_plot(
        exceedance_data,
        thresholds,
        labels,
        colors,
        'Probability of Exceedance',
        'All-hour Precipitation Over Land [mm/h] (Pooled Data)',
        fn,
        percentiles_data
    )
    logger.info(f"Exceedance plot saved: {fn}")


if __name__ == '__main__':
    main(parse_eval_cli())
