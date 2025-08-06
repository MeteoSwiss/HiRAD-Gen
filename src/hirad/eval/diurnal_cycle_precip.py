import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, CONV_FACTOR, WET_THRESHOLD, LOG_INTERVAL

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting computations for diurnal cycle of precipitation amount and wet-hours")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )

    out_root = Path(cfg.generation.io.output_path or './outputs')
    def load(ts, fn):
        return torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR

    # Find channel indices
    indices = get_channel_indices(dataset)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask_da = load_land_sea_mask()
    land_mask = land_mask_da.values
    coords = {"lat": np.arange(land_mask.shape[0]), "lon": np.arange(land_mask.shape[1])}

    # Prepare lists to collect DataArrays
    target_precip, baseline_precip, pred_precip = [], [], []
    target_wet, baseline_wet, pred_wet = [], [], []

    # Collect data
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]
        target = load(ts, f"{ts}-target")[tp_out] * land_mask
        baseline = load(ts, f"{ts}-baseline")[tp_in] * land_mask / 6. # 6 because 1h -> accumulation period is 6h in hourly ERA5 dataset
        preds = load(ts, f"{ts}-predictions")[:, tp_out, :, :] * land_mask

        # DataArrays for spatial means at each timestep
        da_target = xr.DataArray(target, dims=("lat","lon"), coords=coords)
        da_baseline = xr.DataArray(baseline, dims=("lat","lon"), coords=coords)
        da_preds = xr.DataArray(preds, dims=("member","lat","lon"), coords={"member": np.arange(preds.shape[0]), **coords})

        # Spatial mean
        target_precip.append(da_target.mean(dim=("lat","lon")).assign_coords(time=dt))
        baseline_precip.append(da_baseline.mean(dim=("lat","lon")).assign_coords(time=dt))
        pred_precip.append(da_preds.mean(dim=("lat","lon")).assign_coords(time=dt))

        # Wet-hour fraction, i.e., freq(precip) > WET_THRESHOLD
        target_wet.append(((da_target / 24 > WET_THRESHOLD).mean().assign_coords(time=dt)))
        baseline_wet.append(((da_baseline / 24 > WET_THRESHOLD).mean().assign_coords(time=dt)))
        pred_wet.append(((da_preds / 24> WET_THRESHOLD).mean(dim=("lat","lon")).assign_coords(time=dt)))

        if idx % LOG_INTERVAL == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Helper to concat and compute diurnal stats
    def concat_and_group(list_of_da, is_member=False, scale=1.0):
        da = xr.concat(list_of_da, dim="time").groupby("time.hour")
        if is_member:
            timmean = da.mean(dim='time') * scale
            mean = timmean.mean(dim='member')
            std = timmean.std(dim='member')
        else:
            mean = da.mean(dim='time') * scale
            std = None
        return mean, std

    # Compute diurnal means and stds
    amount_target_mean, _ = concat_and_group(target_precip)
    amount_baseline_mean, _ = concat_and_group(baseline_precip)
    amount_pred_mean, amount_pred_std = concat_and_group(pred_precip, is_member=True)

    wet_target_mean, _ = concat_and_group(target_wet, scale=100.0) # scale to percentage
    wet_baseline_mean, _ = concat_and_group(baseline_wet, scale=100.0)
    wet_pred_mean, wet_pred_std = concat_and_group(pred_wet, is_member=True, scale=100.0)

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

    # Generate plots
    save_plot(
        amount_target_mean.hour,
        [amount_target_mean, amount_baseline_mean, amount_pred_mean],
        [None, None, amount_pred_std],
        ['COSMO-2  Analysis','ERA5','CorrDiff ± Std(Members)'],
        'Precipitation (mm/day)',
        'Diurnal Cycle of Precip Amount',
        out_root / 'diurnal_cycle_precip_amount.png'
    )
    save_plot(
        wet_target_mean.hour,
        [wet_target_mean, wet_baseline_mean, wet_pred_mean],
        [None, None, wet_pred_std],
        ['COSMO-2  Analysis','ERA5','CorrDiff ± Std(Members)'],
        'Wet-Hour Fraction [%]',
        'Diurnal Cycle of Wet-Hours (>0.1 mm/h)',
        out_root / 'diurnal_cycle_precip_wethours.png'
    )

    logger.info("Plots saved.")

if __name__ == '__main__':
    main()
