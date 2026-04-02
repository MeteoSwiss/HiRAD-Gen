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
from hirad.eval.eval_utils import resolve_times
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, concat_and_group_diurnal

def main(cfg: dict):
    # Initialize
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

    # Load times
    logger.info("Starting computation for diurnal cycles of 2m temperature and windspeed")
    times = resolve_times(cfg, gen_cfg)
    if times is None:
        logger.error("No times, times_range, or times_ranges specified in config or generation config.")
        return
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Dataset
    dataset_cfg = gen_cfg.get("dataset")
    dataset_type = dataset_cfg.get("type")
    dataset = known_datasets[dataset_type](**dataset_cfg)
    logger.info("Dataset initialized")

    # Indices for channels
    indices = get_channel_indices(dataset)
    out_ch = indices['output']
    in_ch = indices['input']
    
    # Temperature channel (try '2t' first, fallback to 't2m')
    t2m_out = out_ch.get('2t', out_ch.get('t2m'))
    t2m_in = in_ch.get('2t', in_ch.get('t2m', t2m_out))
    
    # Wind channels
    u_out = out_ch['10u']
    u_in = in_ch.get('10u', u_out)
    v_out = out_ch['10v']  
    v_in = in_ch.get('10v', v_out)

    # Output path
    out_root = Path(generation_dir)
    def load(ts, fn):
        return torch.load(out_root/ts/fn, weights_only=False)

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    # Prepare lists to collect DataArrays
    target_temp, baseline_temp, pred_temp, mean_pred_temp = [], [], [], []
    target_wind, baseline_wind, pred_wind, mean_pred_wind = [], [], [], []

    def mean_over_land(data, dims, coords, time_coord):
        da = xr.DataArray(data, dims=dims, coords=coords) * land_mask
        return da.mean(dim=("lat","lon")).assign_coords(time=time_coord)

    # Loop over timestamps
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]

        # Load data
        target = load(ts, f"{ts}-target")
        baseline = load(ts, f"{ts}-baseline")
        predictions = load(ts, f"{ts}-predictions")
        try:
            regression_pred = load(ts, f"{ts}-regression-prediction")
        except:
            regression_pred = None

        # Process temperature (convert to Celsius)
        target_temp.append(mean_over_land(
            target[t2m_out] - 273.15, ("lat","lon"), land_mask.coords, dt))
        baseline_temp.append(mean_over_land(
            baseline[t2m_in] - 273.15, ("lat","lon"), land_mask.coords, dt))
        pred_temp.append(mean_over_land(
            predictions[:, t2m_out, :, :] - 273.15, ("member","lat","lon"), 
            {"member": np.arange(predictions.shape[0]), **land_mask.coords}, dt))
        if regression_pred is not None:
            mean_pred_temp.append(mean_over_land(
                regression_pred[t2m_out] - 273.15, ("lat","lon"), land_mask.coords, dt))


        # Process wind speed
        target_wind.append(mean_over_land(
            np.hypot(target[u_out], target[v_out]), ("lat","lon"), land_mask.coords, dt))
        baseline_wind.append(mean_over_land(
            np.hypot(baseline[u_in], baseline[v_in]), ("lat","lon"), land_mask.coords, dt))
        pred_wind.append(mean_over_land(
            np.hypot(predictions[:, u_out, :, :], predictions[:, v_out, :, :]),
            ("member","lat","lon"), {"member": np.arange(predictions.shape[0]), **land_mask.coords}, dt))
        if regression_pred is not None:
            mean_pred_wind.append(mean_over_land(
                np.hypot(regression_pred[u_out], regression_pred[v_out]), ("lat","lon"), land_mask.coords, dt))

        if idx % cfg.get("log_interval") == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    temp_target_mean, _ = concat_and_group_diurnal(target_temp)
    temp_baseline_mean, _ = concat_and_group_diurnal(baseline_temp)
    temp_pred_mean, temp_pred_std = concat_and_group_diurnal(pred_temp, is_member=True)
    if mean_pred_temp:
        temp_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_temp)

    wind_target_mean, _ = concat_and_group_diurnal(target_wind)
    wind_baseline_mean, _ = concat_and_group_diurnal(baseline_wind)
    wind_pred_mean, wind_pred_std = concat_and_group_diurnal(pred_wind, is_member=True)
    if mean_pred_wind:
        wind_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_wind)

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

    data = [temp_target_mean, temp_baseline_mean, temp_pred_mean, temp_mean_pred_mean] if mean_pred_temp else [temp_target_mean, temp_baseline_mean, temp_pred_mean]
    labels = ['Target', 'Input', 'CorrDiff ± Std(Members)', 'Regression Prediction'] if mean_pred_temp else ['Target', 'Input', 'CorrDiff ± Std(Members)']
    stds = [None, None, temp_pred_std, None] if mean_pred_temp else [None, None, temp_pred_std]

    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    # Generate plots
    save_plot(
        temp_target_mean.hour,
        data,
        stds,
        labels,
        '2m Temperature [°C]',
        'Diurnal Cycle of 2m Temperature',
        output_path / 'diurnal_cycle_2t.png'
    )

    data = [wind_target_mean, wind_baseline_mean, wind_pred_mean, wind_mean_pred_mean] if mean_pred_wind else [wind_target_mean, wind_baseline_mean, wind_pred_mean]
    labels = ['Target', 'Input', 'CorrDiff ± Std(Members)', 'Regression Prediction'] if mean_pred_wind else ['Target', 'Input', 'CorrDiff ± Std(Members)']
    stds = [None, None, wind_pred_std, None] if mean_pred_wind else [None, None, wind_pred_std]

    save_plot(
        wind_target_mean.hour,
        data,
        stds,
        labels,
        'Windspeed [m/s]',
        'Diurnal Cycle of Windspeed',
        output_path / 'diurnal_cycle_windspeed.png'
    )

    logger.info("Plots saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)