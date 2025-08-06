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
from omegaconf import DictConfig, OmegaConf
import xarray as xr

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, CONV_FACTOR_HOURLY, LOG_INTERVAL


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
        colors_perc = {'target': 'blue', 'baseline': 'orange', 'predictions': 'green'}
        legend_added = set()
        
        # Plot all percentile lines
        for dataset_name, data in percentiles_data.items():
            color = colors_perc[dataset_name]
            
            if dataset_name in ['target', 'baseline']:
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

    logger.info("Starting computation for probability of exceedance over land")
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
        return torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR_HOURLY

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask()

    # Define thresholds for exceedance calculation
    thresholds = np.logspace(-2, 2, 200)  # From 0.01 to 100 mm/h
    
    # Storage for exceedance data
    exceedance_data = {}
    # Store all land values for percentile calculation
    all_land_values = {}
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline']:
        logger.info(f"Processing mode: {mode}")
        
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
        
        # Compute exceedance probabilities
        all_values = np.array(all_values)
        exceedance_probs = []
        for threshold in thresholds:
            prob_exceed = np.mean(all_values > threshold)
            exceedance_probs.append(prob_exceed)
        
        exceedance_data[mode] = np.array(exceedance_probs)
        all_land_values[mode] = all_values
        logger.info(f"Processed {len(all_values)} land values for {mode}")
            
    # -- Process predictions: compute exceedance for each ensemble member --
    logger.info("Processing predictions")
    
    n_members = None
    all_member_values = []  # Store all values for each member
    
    for i, ts in enumerate(times):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = load(ts, f"{ts}-predictions")  # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            all_member_values = [[] for _ in range(n_members)]  # Initialize value storage
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
            # Extract land values (remove NaN values)
            land_values = data.values[~np.isnan(data.values)]
            all_member_values[member_idx].extend(land_values)
    
    # Compute exceedance probabilities for each ensemble member
    member_exceedance_data = []
    for member_idx in range(n_members):
        member_values = np.array(all_member_values[member_idx])
        member_exceedance = []
        for threshold in thresholds:
            prob_exceed = np.mean(member_values > threshold)
            member_exceedance.append(prob_exceed)
        member_exceedance_data.append(np.array(member_exceedance))
    
    exceedance_data['predictions'] = tuple(member_exceedance_data)
    
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
    
    
    # Create exceedance plots
    labels = ['COSMO-2 Analysis', 'ERA5', 'CorrDiff Ensemble']
    colors = ['blue', 'orange', 'green']
    
    fn = out_root / 'precipitation_exceedance_over_land.png'
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
    main()
