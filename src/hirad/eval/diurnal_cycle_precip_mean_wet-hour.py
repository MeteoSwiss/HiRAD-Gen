import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.eval.eval_utils import concat_and_group_diurnal, get_channel_indices, load_generation_setup, load_land_sea_mask, parse_eval_cli, precip_conv_factor, resolve_ts_dir

ALLHOUR_THRESHOLDS = [0.1, 1.0, 10.0, 100.0]  # mm/h


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

    logger.info("Starting computations for diurnal cycle of precipitation amount and wet-hours")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    indices = get_channel_indices(gen_cfg)

    # Location of the output from inference
    out_root = Path(generation_dir)

    # Find channel indices
    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))

    conv_factor = precip_conv_factor(cfg)  # mm/h

    # Prepare lists to collect DataArrays
    target_precip, baseline_precip, pred_precip, mean_pred_precip = [], [], [], []
    wet_target   = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_baseline = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_pred     = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_regpred  = {thr: [] for thr in ALLHOUR_THRESHOLDS}

    # Collect data
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]
        target = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-target", weights_only=False)[tp_out] * conv_factor
        baseline = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-baseline", weights_only=False)[tp_in] * conv_factor
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False)[:, tp_out, :, :] * conv_factor
        try:
            mean_pred = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-regression-prediction", weights_only=False)[tp_out] * conv_factor
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

        # Wet-hour fraction per threshold (data already in mm/h)
        for thr in ALLHOUR_THRESHOLDS:
            wet_target[thr].append((da_target > thr).mean().assign_coords(time=dt))
            wet_baseline[thr].append((da_baseline > thr).mean().assign_coords(time=dt))
            wet_pred[thr].append((da_preds > thr).mean(dim=('lat', 'lon')).assign_coords(time=dt))
            if mean_pred is not None:
                wet_regpred[thr].append((da_mean_pred > thr).mean().assign_coords(time=dt))

        if idx % cfg.get("log_interval") == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    amount_target_mean, _ = concat_and_group_diurnal(target_precip)
    amount_baseline_mean, _ = concat_and_group_diurnal(baseline_precip)
    amount_pred_mean, amount_pred_std = concat_and_group_diurnal(pred_precip, is_member=True)
    if mean_pred_precip:
        amount_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_precip)

    # Generate plots
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    save_plot(
        amount_target_mean.hour,
        [amount_target_mean, amount_baseline_mean, amount_pred_mean, amount_mean_pred_mean] if mean_pred_precip else [amount_target_mean, amount_baseline_mean, amount_pred_mean],
        [None, None, amount_pred_std, None] if mean_pred_precip else [None, None, amount_pred_std],
        ['Target','Input','CorrDiff ± Std(Members)', 'Regression Prediction'] if mean_pred_precip else ['Target','Input','CorrDiff ± Std(Members)'],
        'Precipitation (mm/h)',
        'Diurnal Cycle of Precip Amount',
        output_path / 'diurnal_cycle_precip_amount.png'
    )

    # Diurnal cycle of wet-hours, one plot per threshold
    for thr in ALLHOUR_THRESHOLDS:
        wet_target_mean, _ = concat_and_group_diurnal(wet_target[thr], scale=100.0)
        wet_baseline_mean, _ = concat_and_group_diurnal(wet_baseline[thr], scale=100.0)
        wet_pred_mean, wet_pred_std = concat_and_group_diurnal(wet_pred[thr], is_member=True, scale=100.0)
        has_regpred = bool(wet_regpred[thr])
        if has_regpred:
            wet_mean_pred_mean, _ = concat_and_group_diurnal(wet_regpred[thr], scale=100.0)

        fn_wet = output_path / f'diurnal_cycle_precip_wethours_{thr:g}mmh.png'
        save_plot(
            wet_target_mean.hour,
            [wet_target_mean, wet_baseline_mean, wet_pred_mean, wet_mean_pred_mean] if has_regpred else [wet_target_mean, wet_baseline_mean, wet_pred_mean],
            [None, None, wet_pred_std, None] if has_regpred else [None, None, wet_pred_std],
            ['Target','Input','CorrDiff ± Std(Members)', 'Regression Prediction'] if has_regpred else ['Target','Input','CorrDiff ± Std(Members)'],
            'Wet-Hour Fraction [%]',
            f'Diurnal Cycle of Wet-Hours (>{thr:g} mm/h)',
            fn_wet,
        )
        logger.info(f"Diurnal wet-hour plot saved: {fn_wet}")

    logger.info("Plots saved.")

if __name__ == '__main__':
    main(parse_eval_cli())