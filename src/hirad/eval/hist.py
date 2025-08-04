"""
Plots the domain-mean precipitation distribution over land.

This script computes and visualizes the distribution of precipitation values
across the land domain for different data sources (target, baseline, predictions).
"""
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import xarray as xr

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range

# Constants
CONV_FACTOR = 100  # Convert meters to mm/h
LOG_INTERVAL = 24    # Log progress every N timesteps


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
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add percentile lines if provided
    if percentiles_data:        
        # Define colors for different datasets
        percentile_colors = {'target': 'blue', 'baseline': 'orange', 'predictions': 'green'}
        # Track if we've added legend labels for line styles
        legend_added = {'99': False, '99.9': False, '99.99': False}
        
        for dataset_name, percentiles in percentiles_data.items():
            if dataset_name in ['target', 'baseline']:
                # Plot percentiles for target and baseline
                color = percentile_colors[dataset_name]
                for percentile, value in percentiles.items():
                    if percentile == 99:
                        linestyle = '--'
                        legend_label = '99th percentiles' if not legend_added['99'] else None
                        legend_added['99'] = True
                    elif percentile == 99.9:
                        linestyle = ':'
                        legend_label = '99.9th percentiles' if not legend_added['99.9'] else None
                        legend_added['99.9'] = True
                    elif percentile == 99.99:
                        linestyle = '-.'
                        legend_label = '99.99th percentiles' if not legend_added['99.99'] else None
                        legend_added['99.99'] = True
                    
                    plt.vlines(x=value, colors=color, 
                              linestyles=linestyle, alpha=0.8, label=legend_label)
            
            elif dataset_name == 'predictions':
                # Plot percentiles for ensemble members
                color = percentile_colors[dataset_name]
                for member_name, member_percentiles in percentiles.items():
                    for percentile, value in member_percentiles.items():
                        if percentile == 99:
                            linestyle = '--'
                            legend_label = '99th percentiles' if not legend_added['99'] else None
                            legend_added['99'] = True
                        elif percentile == 99.9:
                            linestyle = ':'
                            legend_label = '99.9th percentiles' if not legend_added['99.9'] else None
                            legend_added['99.9'] = True
                        elif percentile == 99.99:
                            linestyle = '-.'
                            legend_label = '99.99th percentiles' if not legend_added['99.99'] else None
                            legend_added['99.99'] = True
                        
                        plt.vlines(x=value, colors=color, 
                                  linestyles=linestyle, alpha=0.6, label=legend_label)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for domain-mean precipitation distribution over land")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Initialize dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    logger.info("Dataset and sampler initialized")

    # Output root and loader
    out_root = Path(cfg.generation.io.output_path or './outputs')
    
    def load(ts, fn):
        return torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR

    # Find channel indices
    out_ch = {c.name: i for i, c in enumerate(dataset.output_channels())}
    in_ch  = {c.name: i for i, c in enumerate(dataset.input_channels())}
    tp_out = out_ch['tp']
    tp_in = in_ch.get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    lsm_data = np.load('/iopsstor/scratch/cscs/davidle/HiRAD-Gen/lsm.npy').reshape(352,544)
    land_mask = xr.DataArray(
        np.where(lsm_data >= 0.5, 1.0, np.nan),
        dims=['lat', 'lon']
    )

    # Define histogram bins
    bins = np.logspace(-1, 1, 50)  # Log-spaced bins for precipitation
    
    # Storage for histogram data
    hist_data = {}
    # Store all land values for percentile calculation
    all_land_values = {}
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline']:
        logger.info(f"Processing mode: {mode}")
        
        # Initialize histogram accumulator and collect all values
        hist_counts = np.zeros(len(bins) - 1)
        total_samples = 0
        all_values = []
        
        for i, ts in enumerate(times):
            if i % LOG_INTERVAL == 0:
                logger.info(f"Processing timestep {i+1}/{len(times)}")
            
            data = load(ts, f"{ts}-{mode}")[tp_out if mode == 'target' else tp_in] * land_mask
            
            # Apply scaling factor for baseline
            if mode == 'baseline':
                data = data / 6.0
            
            # Extract land values (remove NaN values)
            land_values = data.values[~np.isnan(data.values)]
            all_values.extend(land_values)
            
            # Accumulate histogram counts
            counts, _ = np.histogram(land_values, bins=bins)
            hist_counts += counts
            total_samples += len(land_values)
        
        # Normalize to probability density
        bin_widths = np.diff(bins)
        hist_data[mode] = hist_counts / (total_samples * bin_widths)
        all_land_values[mode] = np.array(all_values)
        logger.info(f"Processed {total_samples} land values for {mode}")
            
    # -- Process predictions: compute histogram for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    member_hist_data = []
    all_member_values = []  # Store all values for each member
    
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = load(ts, f"{ts}-predictions")  # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            # Initialize histogram accumulators for each member
            member_hist_data = [np.zeros(len(bins) - 1) for _ in range(n_members)]
            member_sample_counts = [0 for _ in range(n_members)]
            all_member_values = [[] for _ in range(n_members)]  # Initialize value storage
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
            # Extract land values (remove NaN values)
            land_values = data.values[~np.isnan(data.values)]
            all_member_values[member_idx].extend(land_values)  # Store values for percentiles
            
            # Accumulate histogram counts for this member
            counts, _ = np.histogram(land_values, bins=bins)
            member_hist_data[member_idx] += counts
            member_sample_counts[member_idx] += len(land_values)
    
    # Normalize member histograms to probability density
    bin_widths = np.diff(bins)
    normalized_member_hists = []
    for member_idx in range(n_members):
        normalized_hist = member_hist_data[member_idx] / (member_sample_counts[member_idx] * bin_widths)
        normalized_member_hists.append(normalized_hist)
    
    hist_data['predictions'] = tuple(normalized_member_hists)
    
    logger.info(f"Collected {n_members} ensemble members for predictions")
    
    # Compute percentiles for all datasets
    percentiles_data = {}
    
    # Target percentiles
    target_data_array = xr.DataArray(all_land_values['target'])
    target_p99 = target_data_array.quantile(0.99).item()
    target_p999 = target_data_array.quantile(0.999).item()
    target_p9999 = target_data_array.quantile(0.9999).item()
    percentiles_data['target'] = {99: target_p99, 99.9: target_p999, 99.99: target_p9999}
    
    # Baseline percentiles
    baseline_data_array = xr.DataArray(all_land_values['baseline'])
    baseline_p99 = baseline_data_array.quantile(0.99).item()
    baseline_p999 = baseline_data_array.quantile(0.999).item()
    baseline_p9999 = baseline_data_array.quantile(0.9999).item()
    percentiles_data['baseline'] = {99: baseline_p99, 99.9: baseline_p999, 99.99: baseline_p9999}
    
    # Ensemble member percentiles
    percentiles_data['predictions'] = {}
    for member_idx in range(n_members):
        member_data_array = xr.DataArray(all_member_values[member_idx])
        member_p99 = member_data_array.quantile(0.99).item()
        member_p999 = member_data_array.quantile(0.999).item()
        member_p9999 = member_data_array.quantile(0.9999).item()
        percentiles_data['predictions'][f'member_{member_idx}'] = {99: member_p99, 99.9: member_p999, 99.99: member_p9999}
    
    
    # Create distribution plots
    labels = ['COSMO-2', 'ERA5', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'green']
    
    fn = out_root / 'precipitation_distribution_over_land.png'
    save_distribution_plot(
        hist_data,
        bins,
        labels,
        colors,
        'Domain-Mean Precip. Over Land (Pooled Data)',
        'Precipitation (mm/h)',
        fn,
        percentiles_data
    )
    logger.info(f"Distribution plot saved: {fn}")


if __name__ == '__main__':
    main()