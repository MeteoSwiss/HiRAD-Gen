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
from omegaconf import DictConfig, OmegaConf
import xarray as xr

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, CONV_FACTOR_HOURLY, LOG_INTERVAL


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

    # Output root
    out_root = Path(cfg.generation.io.output_path or './outputs')

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask()

    # Define histogram bins
    bins = np.logspace(-1, 3.3, 200)  # Log-spaced bins for precipitation
    
    # Storage for histogram data and land values
    hist_data = {}
    all_land_values = {}
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        hist_counts = np.zeros(len(bins) - 1)
        total_samples = 0
        all_values = []
        
        try:
            for i, ts in enumerate(times):
                if i % LOG_INTERVAL == 0:
                    logger.info(f"Processing timestep {i+1}/{len(times)}")
                
                data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False)[tp_out if mode in ['target', 'regression-prediction'] else tp_in] * CONV_FACTOR_HOURLY * land_mask
                
                # Apply scaling factor for baseline
                # if mode == 'baseline':
                #     data = data / 6.0
                
                land_values = data.values[~np.isnan(data.values)]
                all_values.extend(land_values)
                
                counts, _ = np.histogram(land_values, bins=bins)
                hist_counts += counts
                total_samples += len(land_values)
        except:
            logger.warning(f"{mode} not available, skipping")
            continue        
        # Normalize to probability density
        bin_widths = np.diff(bins)
        hist_data[mode] = hist_counts / (total_samples * bin_widths)
        all_land_values[mode] = np.array(all_values)
        logger.info(f"Processed {total_samples} land values for {mode}")
            
    # -- Process predictions: compute histogram for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    member_hist_data = []
    all_member_values = []
    
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False) * CONV_FACTOR_HOURLY  # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            member_hist_data = [np.zeros(len(bins) - 1) for _ in range(n_members)]
            member_sample_counts = [0 for _ in range(n_members)]
            all_member_values = [[] for _ in range(n_members)]
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
            land_values = data.values[~np.isnan(data.values)]
            all_member_values[member_idx].extend(land_values)
            
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
    percentiles = {99: 0.99, 99.9: 0.999, 99.99: 0.9999}
    
    # Target and baseline percentiles
    for mode in ['target', 'baseline', 'regression-prediction']:
        if mode in all_land_values:
            data_array = xr.DataArray(all_land_values[mode])
            percentiles_data[mode] = {
                key: data_array.quantile(p).item() 
                for key, p in percentiles.items()
            }
    
    # Ensemble member percentiles
    percentiles_data['predictions'] = {}
    for member_idx in range(n_members):
        member_data_array = xr.DataArray(all_member_values[member_idx])
        percentiles_data['predictions'][f'member_{member_idx}'] = {
            key: member_data_array.quantile(p).item()
            for key, p in percentiles.items()
        }
    
    # Create distribution plots
    labels = ['COSMO-2  Analysis', 'ERA5', 'Regression Prediction', 'CorrDiff Ensemble'] if 'regression-prediction' in hist_data else ['COSMO-2  Analysis', 'ERA5', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'red', 'green'] if 'regression-prediction' in hist_data else ['blue', 'orange', 'green']
    
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