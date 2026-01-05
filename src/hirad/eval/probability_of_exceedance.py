"""
Plots the probability of exceedance for precipitation over land.

This script computes and visualizes the complementary cumulative distribution
(probability of exceeding x mm/h) over land).
"""
import logging
import argparse
import yaml
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask


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

    generation_dir = cfg.get("inference_output_dir", None)
    if generation_dir is None:
        logger.error("No inference_output_dir specified in config.")
        return
    
    if not Path(generation_dir).exists() or not Path(generation_dir).is_dir():
        logger.error(f"Inference output directory {generation_dir} does not exist or is not a directory.")
        return

    generation_config_path = Path(generation_dir) / ".hydra" / "config.yaml"
    if not generation_config_path.exists():
        logger.error(f"Generation config file {generation_config_path} does not exist.")
        return

    with open(generation_config_path, "r") as f:
        gen_cfg = yaml.safe_load(f)

    logger.info("Starting computation for probability of exceedance over land")
    if cfg.get("times_range", None):
        times = get_time_from_range(cfg.get("times_range"), time_format="%Y%m%d-%H%M")
    elif cfg.get("times", None):
        times = cfg.get("times")
    elif gen_cfg.get("generation").get("times_range", None):
        times = get_time_from_range(gen_cfg.get("generation").get("times_range"), time_format="%Y%m%d-%H%M")
    elif gen_cfg.get("generation").get("times", None):
        times = gen_cfg.get("generation").get("times")
    else:
        logger.error("No times or times_range specified in config or generation config.")
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Initialize dataset
    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.pop("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    # Output root
    out_root = Path(generation_dir)

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    # Define thresholds for exceedance calculation
    thresholds = np.logspace(-2, 2.1, 200)  # From 0.01 to 100 mm/h
    
    # Storage for exceedance data and land values
    exceedance_data = {}
    all_land_values = {}
    
    # -- Process target and baseline --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        
        all_values = []
        
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"Processing timestep {i+1}/{len(times)}")
                
                data = torch.load(out_root/ts/f"{ts}-{mode}", weights_only=False)[tp_out if mode in ['target','regression-prediction'] else tp_in] * cfg.get("conv_factor_hourly") * land_mask
                
                land_values = data.values[~np.isnan(data.values)]
                all_values.extend(land_values)
        except:
            logger.warning(f"{mode} data not found, skipping")
            continue

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
    all_member_values = []
    
    for i, ts in enumerate(times):
        if i % cfg.get("log_interval") == 0:
            logger.info(f"Processing timestep {i+1}/{len(times)}")
        
        preds = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False) * cfg.get("conv_factor_hourly") # [n_members, n_channels, lat, lon]
        
        if n_members is None:
            n_members = preds.shape[0]
            all_member_values = [[] for _ in range(n_members)]
        
        for member_idx in range(n_members):
            data = preds[member_idx, tp_out] * land_mask
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
