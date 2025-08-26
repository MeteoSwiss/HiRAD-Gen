import torch
import yaml
import numpy as np
import os
from pathlib import Path
import argparse
from hirad.eval import plotting
from hirad.utils.inference_utils import calculate_bounds, transform_channel
import hydra
from omegaconf import DictConfig, OmegaConf

channel_plot_args = {
    "2t": {"label": "°C"},
    "10u": {"label": "m/s"},
    "10v": {"label": "m/s"},
    "tp": {"label": "boxcox(mm/h)"},
}


def get_available_time_steps(results_dir):
    """Find all available time step directories"""
    results_path = Path(results_dir)
    time_step_dirs = [d.name for d in results_path.iterdir() if d.is_dir() and not d.name.startswith('plots') and not d.name.startswith('maps')]
    return sorted(time_step_dirs)

def plot_time_step(results_dir, output_dir, time_step, input_channels, output_channels, cfg):
    """Plot all channels for a single time step"""
    print(f"Processing time step: {time_step}")
    
    # Set up paths for this time step
    ts_results_dir = Path(results_dir) / time_step
    
    # Load tensors
    try:
        target = torch.load(ts_results_dir / f"{time_step}-target", weights_only=False)
        baseline = torch.load(ts_results_dir / f"{time_step}-baseline", weights_only=False)
        predictions = torch.load(ts_results_dir / f"{time_step}-predictions", weights_only=False)
        
        try:
            mean_pred = torch.load(ts_results_dir / f"{time_step}-regression-prediction", weights_only=False)
        except FileNotFoundError:
            mean_pred = None
            print(f"  Warning: No mean prediction found for {time_step}")
            
    except FileNotFoundError as e:
        print(f"  Error: Missing required file for {time_step}: {e}")
        return

    # Create output directory for this time step
    ts_output_dir = Path(output_dir) / time_step
    ts_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot each channel
    for idx, channel in enumerate(output_channels):
        print(f"  Plotting channel: {channel}")
        
        # Get input channel index (handle case where input/output channels differ)
        try:
            input_idx = input_channels.index(channel)
        except ValueError:
            print(f"    Warning: Channel {channel} not in input channels, skipping baseline")
            continue
            
        # Transform data
        if channel != "tp" or not cfg.get("plot_box_precipitation", False):
            tgt = transform_channel(target[idx], channel)
            base = transform_channel(baseline[input_idx], channel)
            preds = transform_channel(predictions[:, idx], channel)
            mean = transform_channel(mean_pred[idx], channel) if mean_pred is not None else None
            if channel == "tp":
                threshold = transform_channel(np.array([cfg.get("tp_threshold", 0.002)]), "tp")[0]  # Transform threshold too
                tgt = np.ma.masked_where(tgt <= threshold, tgt)
                base = np.ma.masked_where(base <= threshold, base)
                preds = np.ma.masked_where(preds <= threshold, preds)
                if mean is not None:
                    mean = np.ma.masked_where(mean <= threshold, mean)
        else:
            # For precipitation, use raw values if plotting box precipitation
            tgt = target[idx]
            base = baseline[input_idx]
            preds = predictions[:, idx]
            mean = mean_pred[idx] if mean_pred is not None else None

        # Calculate consistent bounds (skip for precipitation)
        if channel != "tp" or not cfg.get("plot_box_precipitation", False):
            arrays = [tgt, base] + [preds[i] for i in range(preds.shape[0])]
            if mean is not None:
                arrays.append(mean)
            vmin, vmax = calculate_bounds(*arrays)
        else:
            vmin, vmax = None, None

        base_channel_dir = ts_output_dir / channel
        base_channel_dir.mkdir(parents=True, exist_ok=True)

        # Plot target
        fname = ts_output_dir / channel / f"target"
        if channel == "tp" and cfg.get("plot_box_precipitation", False):
            plotting.plot_map_precipitation(tgt, str(fname), title=f"Target - {channel}")
        else:
            plotting.plot_map(tgt, str(fname), vmin=vmin, vmax=vmax, 
                            title=f"Target - {channel}", **channel_plot_args.get(channel, {}))

        # Plot baseline
        fname = ts_output_dir / channel / "baseline"
        if channel == "tp" and cfg.get("plot_box_precipitation", False):
            plotting.plot_map_precipitation(base, str(fname), title=f"Baseline - {channel}")
        else:
            plotting.plot_map(base, str(fname), vmin=vmin, vmax=vmax, 
                            title=f"Baseline - {channel}", **channel_plot_args.get(channel, {}))

        # Plot mean prediction if available
        if mean is not None:
            fname = ts_output_dir / channel / "mean-prediction"
            if channel == "tp" and cfg.get("plot_box_precipitation", False):
                plotting.plot_map_precipitation(mean, str(fname), title=f"Mean Prediction - {channel}")
            else:
                plotting.plot_map(mean, str(fname), vmin=vmin, vmax=vmax, 
                                title=f"Mean Prediction - {channel}", **channel_plot_args.get(channel, {}))

        # Plot ensemble members
        for member_idx in range(preds.shape[0]):
            fname = ts_output_dir / channel / f"prediction_{member_idx:02d}"
            if channel == "tp" and cfg.get("plot_box_precipitation", False):
                plotting.plot_map_precipitation(
                    preds[member_idx], str(fname), 
                    title=f"Prediction {member_idx} - {channel}"
                )
            else:
                plotting.plot_map(
                    preds[member_idx], str(fname), 
                    vmin=vmin, vmax=vmax,
                    title=f"Prediction {member_idx} - {channel}",
                    **channel_plot_args.get(channel, {})
                )

@hydra.main(version_base=None, config_path="../conf", config_name="plotting")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    
    input_channels = cfg.dataset.input_channel_names
    output_channels = cfg.dataset.output_channel_names
    
    # Set up directories
    results_dir = Path(cfg.results_dir)
    output_dir = Path(cfg.output_dir) if "output_dir" in cfg and cfg.output_dir else results_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Determine time steps to process
    if cfg.time_steps:
        time_steps = cfg.time_steps
    else:
        time_steps = get_available_time_steps(results_dir)
        print(f"Found {len(time_steps)} time steps: {time_steps}")
    
    # Process each time step
    for time_step in time_steps:
        plot_time_step(results_dir, output_dir, time_step, input_channels, output_channels, cfg)
    
    print(f"Plotting complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()