import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.eval.eval_utils import concat_and_group_diurnal, get_channel_indices, load_generation_setup, load_land_sea_mask, parse_eval_cli, resolve_ts_dir

def main(cfg: dict):
    # Initialize
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computation for diurnal cycle of 2m temperature")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Indices for channels
    indices = get_channel_indices(gen_cfg)
    out_ch = indices['output']
    in_ch = indices['input']

    # Temperature channel (try '2t' first, fallback to 't2m')
    t2m_out = out_ch.get('2t', out_ch.get('t2m'))
    t2m_in = in_ch.get('2t', in_ch.get('t2m', t2m_out))

    # Output path
    out_root = Path(generation_dir)
    def load(ts, fn):
        return torch.load(resolve_ts_dir(out_root, ts) / ts / fn, weights_only=False)

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    # Prepare lists to collect DataArrays
    target_temp, baseline_temp, pred_temp, mean_pred_temp = [], [], [], []

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

        if idx % cfg.get("log_interval") == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    temp_target_mean, _ = concat_and_group_diurnal(target_temp)
    temp_baseline_mean, _ = concat_and_group_diurnal(baseline_temp)
    temp_pred_mean, temp_pred_std = concat_and_group_diurnal(pred_temp, is_member=True)
    if mean_pred_temp:
        temp_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_temp)

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
    save_plot(
        temp_target_mean.hour,
        data,
        stds,
        labels,
        '2m Temperature [°C]',
        'Diurnal Cycle of 2m Temperature',
        output_path / 'diurnal_cycle_2t.png'
    )

    logger.info("Plots saved.")

if __name__ == '__main__':
    main(parse_eval_cli())
