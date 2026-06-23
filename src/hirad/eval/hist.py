"""
Plots the domain-mean precipitation distribution over land.

This script computes and visualizes the distribution of precipitation values
over land.
"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from hirad.eval.eval_utils import get_channel_indices, load_generation_setup, load_land_sea_mask, parse_eval_cli, precip_conv_factor, resolve_ts_dir
from hirad.eval.eval_utils import percentiles_from_histogram


def save_distribution_plot(hist_data_dict, bin_edges, labels, colors, title, ylabel, out_path, percentiles_data=None):
    """Save distribution plot with pre-computed histograms."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms from pre-computed bin counts
    for (key, hist_data), label, color in zip(hist_data_dict.items(), labels, colors):
        if isinstance(hist_data, tuple):  # Handle ensemble data
            # Plot individual members with transparency
            for i, member_hist in enumerate(hist_data):
                alpha = 0.5 if i > 0 else 0.7
                label_member = label if i == 0 else None
                # Plot histogram from bin counts
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, member_hist, alpha=alpha, color=color, 
                        label=label_member, drawstyle='steps-mid')
        else:
            # Plot histogram from bin counts
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, hist_data, alpha=0.7, color=color, 
                    label=label, linewidth=2, drawstyle='steps-mid')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(ylabel)
    plt.ylabel('Probability Density')
    plt.ylim(1e-8, 1)
    plt.xlim(bin_edges[1], bin_edges[-1])
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
        colors = {'target': 'blue', 'baseline': 'orange', 'predictions': 'green', 'regression-prediction': 'red'}
        legend_added = set()
        
        # Plot all percentile lines
        for dataset_name, data in percentiles_data.items():
            color = colors[dataset_name]
            
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

    logger.info("Starting computation for domain-mean precipitation distribution over land")
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

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    conv_factor = precip_conv_factor(cfg)  # mm/h

    # Define histogram bins
    # bins = np.logspace(-1, 3.3, 200)  # Log-spaced bins for precipitation
    log_bins = np.logspace(-1, 3.3, 200)  # Log-spaced bins for precipitation
    bins = np.concatenate([[0], log_bins])  # Prepend 0 to capture all sub-0.1 values

    # Storage for histogram data and land values
    hist_data = {}
    raw_hist_counts = {}  # Store raw counts for percentile estimation
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        hist_counts = np.zeros(len(bins) - 1)
        total_samples = 0
        
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"Processing timestep {i+1}/{len(times)}")
                
                data = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-{mode}", weights_only=False)[tp_out if mode in ['target', 'regression-prediction'] else tp_in] * conv_factor * land_mask
                
                land_values = data.values[~np.isnan(data.values)]
                
                counts, _ = np.histogram(land_values, bins=bins)
                hist_counts += counts
                total_samples += len(land_values)
        except:
            logger.warning(f"{mode} not available, skipping")
            continue
        # Store raw counts for percentile estimation
        raw_hist_counts[mode] = hist_counts.copy()
        # Normalize to probability density
        bin_widths = np.diff(bins)
        hist_data[mode] = hist_counts[1:] / (total_samples * bin_widths[1:])
        logger.info(f"Processed {total_samples} land values for {mode}")
            
    # -- Process predictions: compute histogram for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    member_hist_data = []
    
    for i, ts in enumerate(times):
        if i % cfg.get("log_interval") == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False) * conv_factor  # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            member_hist_data = [np.zeros(len(bins) - 1) for _ in range(n_members)]
            member_sample_counts = [0 for _ in range(n_members)]
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
            land_values = data.values[~np.isnan(data.values)]
            
            counts, _ = np.histogram(land_values, bins=bins)
            member_hist_data[member_idx] += counts
            member_sample_counts[member_idx] += len(land_values)
    
    # Normalize member histograms to probability density
    bin_widths = np.diff(bins)
    normalized_member_hists = []
    for member_idx in range(n_members):
        normalized_hist = member_hist_data[member_idx][1:] / (member_sample_counts[member_idx] * bin_widths[1:])
        normalized_member_hists.append(normalized_hist)
    
    hist_data['predictions'] = tuple(normalized_member_hists)
    
    logger.info(f"Collected {n_members} ensemble members for predictions")
    
    # Compute percentiles for all datasets
    percentiles_data = {}
    percentiles = {99: 0.99, 99.9: 0.999, 99.99: 0.9999}
    
    # Target and baseline percentiles
    for mode in ['target', 'baseline', 'regression-prediction']:
        if mode in raw_hist_counts:
            cumulative = np.cumsum(raw_hist_counts[mode])
            total = cumulative[-1]         
            cdf = cumulative / total  # CDF at upper bin edges
            percentiles_data[mode] = percentiles_from_histogram(
                raw_hist_counts[mode], bins, percentiles
            )
    
    # Ensemble member percentiles
    percentiles_data['predictions'] = {}
    for member_idx in range(n_members):
        cumulative = np.cumsum(member_hist_data[member_idx])
        total = cumulative[-1]
        cdf = cumulative / total  # CDF at upper bin edges
        percentiles_data['predictions'][f'member_{member_idx}'] = percentiles_from_histogram(
            member_hist_data[member_idx], bins, percentiles
        )
    
    # Create distribution plots
    labels = ['Target', 'Input', 'Regression Prediction', 'CorrDiff Ensemble'] if 'regression-prediction' in hist_data else ['Target', 'Input', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'red', 'green'] if 'regression-prediction' in hist_data else ['blue', 'orange', 'green']
    
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    fn = output_path / 'precipitation_distribution_over_land.png'
    save_distribution_plot(
        hist_data,  # Skip the first bin (0 to 0.1) for plotting
        bins[1:],
        labels,
        colors,
        'Domain-Mean Precip. Over Land (Pooled Data)',
        'Precipitation (mm/h)',
        fn,
        percentiles_data
    )
    logger.info(f"Distribution plot saved: {fn}")


if __name__ == '__main__':
    main(parse_eval_cli())