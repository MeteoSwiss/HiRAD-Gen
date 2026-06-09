import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from hirad.eval.eval_utils import concat_and_group_diurnal, get_channel_indices, load_generation_setup, load_land_sea_mask, parse_eval_cli, resolve_ts_dir

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


def save_allhour_wethours_plot(thresholds, mode_data, labels, title, out_path):
    """Bar chart of all-hour wet-hour fraction (%) per threshold.

    mode_data: list of dicts {thr: (mean_pct, std_pct_or_None)}
    """
    n_thr = len(thresholds)
    n_modes = len(mode_data)
    x = np.arange(n_thr)
    width = 0.7 / n_modes
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (data, label) in enumerate(zip(mode_data, labels)):
        offset = (i - n_modes / 2 + 0.5) * width
        means = [data[thr][0] for thr in thresholds]
        stds  = [data[thr][1] if data[thr][1] is not None else 0.0 for thr in thresholds]
        ax.bar(x + offset, means, width, label=label, alpha=0.8)
        if any(s > 0 for s in stds):
            ax.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'>{thr:g} mm/h' for thr in thresholds])
    ax.set_ylabel('Wet-Hour Fraction [%]')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, which='both')
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

    # Prepare lists to collect DataArrays
    target_precip, baseline_precip, pred_precip, mean_pred_precip = [], [], [], []
    wet_target   = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_baseline = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_pred     = {thr: [] for thr in ALLHOUR_THRESHOLDS}
    wet_regpred  = {thr: [] for thr in ALLHOUR_THRESHOLDS}

    # Collect data
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]
        target = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-target", weights_only=False)[tp_out] * cfg.get("conv_factor")
        baseline = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-baseline", weights_only=False)[tp_in] * cfg.get("conv_factor")
        preds = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-predictions", weights_only=False)[:, tp_out, :, :] * cfg.get("conv_factor")
        try:
            mean_pred = torch.load(resolve_ts_dir(out_root, ts)/ts/f"{ts}-regression-prediction", weights_only=False)[tp_out] * cfg.get("conv_factor")
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

        # Wet-hour fraction per threshold
        for thr in ALLHOUR_THRESHOLDS:
            wet_target[thr].append((da_target / 24 > thr).mean().assign_coords(time=dt))
            wet_baseline[thr].append((da_baseline / 24 > thr).mean().assign_coords(time=dt))
            wet_pred[thr].append((da_preds / 24 > thr).mean(dim=('lat', 'lon')).assign_coords(time=dt))
            if mean_pred is not None:
                wet_regpred[thr].append((da_mean_pred / 24 > thr).mean().assign_coords(time=dt))

        if idx % cfg.get("log_interval") == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    amount_target_mean, _ = concat_and_group_diurnal(target_precip)
    amount_baseline_mean, _ = concat_and_group_diurnal(baseline_precip)
    amount_pred_mean, amount_pred_std = concat_and_group_diurnal(pred_precip, is_member=True)
    if mean_pred_precip:
        amount_mean_pred_mean, _ = concat_and_group_diurnal(mean_pred_precip)

    wet_target_mean, _ = concat_and_group_diurnal(wet_target[0.1], scale=100.0)
    wet_baseline_mean, _ = concat_and_group_diurnal(wet_baseline[0.1], scale=100.0)
    wet_pred_mean, wet_pred_std = concat_and_group_diurnal(wet_pred[0.1], is_member=True, scale=100.0)
    if wet_regpred[0.1]:
        wet_mean_pred_mean, _ = concat_and_group_diurnal(wet_regpred[0.1], scale=100.0)

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
        [wet_target_mean, wet_baseline_mean, wet_pred_mean, wet_mean_pred_mean] if wet_regpred[0.1] else [wet_target_mean, wet_baseline_mean, wet_pred_mean],
        [None, None, wet_pred_std, None] if wet_regpred[0.1] else [None, None, wet_pred_std],
        ['Target','Input','CorrDiff ± Std(Members)', 'Regression Prediction'] if wet_regpred[0.1] else ['Target','Input','CorrDiff ± Std(Members)'],
        'Wet-Hour Fraction [%]',
        'Diurnal Cycle of Wet-Hours (>0.1 mm/h)',
        output_path / 'diurnal_cycle_precip_wethours.png'
    )

    # All-hour wet-hour fraction bar chart
    pred_all = {thr: xr.concat(wet_pred[thr], dim='time').values.ravel() for thr in ALLHOUR_THRESHOLDS}
    allhour_mode_data = [
        {thr: (float(xr.concat(wet_target[thr],   dim='time').mean()) * 100, None) for thr in ALLHOUR_THRESHOLDS},
        {thr: (float(xr.concat(wet_baseline[thr], dim='time').mean()) * 100, None) for thr in ALLHOUR_THRESHOLDS},
        {thr: (float(pred_all[thr].mean()) * 100, float(pred_all[thr].std()) * 100) for thr in ALLHOUR_THRESHOLDS},
    ]
    allhour_labels = ['Target', 'Input', 'CorrDiff ± Std(Members)']
    if any(wet_regpred[thr] for thr in ALLHOUR_THRESHOLDS):
        allhour_mode_data.append(
            {thr: (float(xr.concat(wet_regpred[thr], dim='time').mean()) * 100, None) for thr in ALLHOUR_THRESHOLDS}
        )
        allhour_labels.append('Regression Prediction')
    fn_allhour = output_path / 'allhour_wethours.png'
    save_allhour_wethours_plot(
        ALLHOUR_THRESHOLDS, allhour_mode_data, allhour_labels,
        'All-Hour Wet-Hour Fraction', fn_allhour,
    )
    logger.info(f"All-hour wet-hour plot saved: {fn_allhour}")

    logger.info("Plots saved.")

if __name__ == '__main__':
    main(parse_eval_cli())