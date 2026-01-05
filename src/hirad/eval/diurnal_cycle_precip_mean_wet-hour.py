import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, concat_and_group_diurnal

def save_plot(hour, means, stds, labels, ylabel, title, out_path):
    hrs = np.concatenate([hour.values, [24]])
    plt.figure(figsize=(8,4))
    for mean, std, label in zip(means, stds, labels):
        vals = np.append(mean.values, mean.values[0])
        line, = plt.plot(hrs, vals, label=label)
        if std is not None:
            stdv = np.append(std.values, std.values[0])
            plt.fill_between(hrs, np.maximum(vals - stdv, 0), vals + stdv, color=line.get_color(), alpha=0.3)
    plt.xlabel('Hour (UTC)')
    plt.xticks(range(0,25,3))
    plt.xlim(0,24)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
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

    logger.info("Starting computations for diurnal cycle of precipitation amount and wet-hours")
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
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.pop("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    # Location of the output from inference
    out_root = Path(generation_dir)

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    # Prepare lists to collect DataArrays
    target_precip, baseline_precip, pred_precip, mean_pred_precip = [], [], [], []
    target_wet, baseline_wet, pred_wet, mean_pred_wet = [], [], [], []

    # Collect data
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]
        target = torch.load(out_root/ts/f"{ts}-target", weights_only=False)[tp_out] * cfg.get("conv_factor")
        baseline = torch.load(out_root/ts/f"{ts}-baseline", weights_only=False)[tp_in] * cfg.get("conv_factor")
        preds = torch.load(out_root/ts/f"{ts}-predictions", weights_only=False)[:, tp_out, :, :] * cfg.get("conv_factor")
        try:
            mean_pred = torch.load(out_root/ts/f"{ts}-regression-prediction", weights_only=False)[tp_out] * cfg.get("conv_factor")
        except:
            mean_pred = None

        # DataArrays for spatial means at each timestep
        da_target = xr.DataArray(target, dims=("lat","lon"), coords=land_mask.coords)
        da_baseline = xr.DataArray(baseline, dims=("lat","lon"), coords=land_mask.coords)
        da_preds = xr.DataArray(preds, dims=("member","lat","lon"), coords={"member": np.arange(preds.shape[0]), **land_mask.coords})
        if mean_pred is not None:
            da_mean_pred = xr.DataArray(mean_pred, dims=("lat","lon"), coords=land_mask.coords)

        # Apply land mask after conversion to xarray
        da_target = da_target * land_mask
        da_baseline = da_baseline * land_mask
        da_preds = da_preds * land_mask
        if mean_pred is not None:
            da_mean_pred = da_mean_pred * land_mask

        # Spatial mean
        target_precip.append(da_target.mean(dim=("lat","lon")).assign_coords(time=dt))
        baseline_precip.append(da_baseline.mean(dim=("lat","lon")).assign_coords(time=dt))
        pred_precip.append(da_preds.mean(dim=("lat","lon")).assign_coords(time=dt))
        if mean_pred is not None:
            mean_pred_precip.append(da_mean_pred.mean(dim=("lat","lon")).assign_coords(time=dt))

        # Wet-hour fraction, i.e., freq(precip) > wet_threshold
        target_wet.append(((da_target / 24 > cfg.get("wet_threshold")).mean().assign_coords(time=dt)))
        baseline_wet.append(((da_baseline / 24 > cfg.get("wet_threshold")).mean().assign_coords(time=dt)))
        pred_wet.append(((da_preds / 24> cfg.get("wet_threshold")).mean(dim=("lat","lon")).assign_coords(time=dt)))
        if mean_pred is not None:
            mean_pred_wet.append(((da_mean_pred / 24 > cfg.get("wet_threshold")).mean().assign_coords(time=dt)))

        if idx % cfg.get("log_interval") == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    amount_target_mean, _ = concat_and_group_diurnal(target_precip)
    amount_baseline_mean, _ = concat_and_group_diurnal(baseline_precip)
    amount_pred_mean, amount_pred_std = concat_and_group_diurnal(pred_precip, is_member=True)
    if mean_pred_precip:
        amount_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_precip)

    wet_target_mean, _ = concat_and_group_diurnal(target_wet, scale=100.0) # scale to obtain percentages
    wet_baseline_mean, _ = concat_and_group_diurnal(baseline_wet, scale=100.0)
    wet_pred_mean, wet_pred_std = concat_and_group_diurnal(pred_wet, is_member=True, scale=100.0)
    if mean_pred_wet:
        wet_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_wet, scale=100.0)

    # Generate plots
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    save_plot(
        amount_target_mean.hour,
        [amount_target_mean, amount_baseline_mean, amount_pred_mean, amount_mean_pred_mean] if mean_pred_precip else [amount_target_mean, amount_baseline_mean, amount_pred_mean],
        [None, None, amount_pred_std, None] if mean_pred_precip else [None, None, amount_pred_std],
        ['Target','Input','CorrDiff ± Std(Members)', 'Regression Prediction'] if mean_pred_precip else ['Target','Input','CorrDiff ± Std(Members)'],
        'Precipitation (mm/day)',
        'Diurnal Cycle of Precip Amount',
        output_path / 'diurnal_cycle_precip_amount.png'
    )
    save_plot(
        wet_target_mean.hour,
        [wet_target_mean, wet_baseline_mean, wet_pred_mean, wet_mean_pred_mean] if mean_pred_wet else [wet_target_mean, wet_baseline_mean, wet_pred_mean],
        [None, None, wet_pred_std, None] if mean_pred_wet else [None, None, wet_pred_std],
        ['Target','Input','CorrDiff ± Std(Members)', 'Regression Prediction'] if mean_pred_wet else ['Target','Input','CorrDiff ± Std(Members)'],
        'Wet-Hour Fraction [%]',
        'Diurnal Cycle of Wet-Hours (>0.1 mm/h)',
        output_path / 'diurnal_cycle_precip_wethours.png'
    )

    logger.info("Plots saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)