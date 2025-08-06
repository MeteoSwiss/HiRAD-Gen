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
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask, LOG_INTERVAL, concat_and_group_diurnal

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Initialize
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load times
    logger.info("Starting computation for diurnal cycles of 2m temperature and windspeed")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )

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
    out_root = Path(cfg.generation.io.output_path or './outputs')
    def load(ts, fn):
        return torch.load(out_root/ts/fn, weights_only=False)

    # Land-sea mask
    land_mask_da = load_land_sea_mask()
    land_mask = land_mask_da.values
    coords = {"lat": np.arange(land_mask.shape[0]), "lon": np.arange(land_mask.shape[1])}

    # Prepare lists to collect DataArrays
    target_temp, baseline_temp, pred_temp = [], [], []
    target_wind, baseline_wind, pred_wind = [], [], []

    # Loop over timestamps
    for idx, ts in enumerate(times, 1):
        dt = datetimes[idx-1]

        # Load and apply land mask
        target = load(ts, f"{ts}-target") * land_mask
        baseline = load(ts, f"{ts}-baseline") * land_mask
        predictions = load(ts, f"{ts}-predictions") * land_mask

        # Wrap into DataArrays (convert temperature to Celsius inline)
        da_tgt_temp = xr.DataArray(
            target[t2m_out] - 273.15, dims=("lat","lon"), coords=coords
        )
        da_bsl_temp = xr.DataArray(
            baseline[t2m_in] - 273.15, dims=("lat","lon"), coords=coords
        )
        tgt_wind = np.hypot(target[u_out], target[v_out])
        bsl_wind = np.hypot(baseline[u_in], baseline[v_in])
        da_tgt_wind = xr.DataArray(tgt_wind, dims=("lat","lon"), coords=coords)
        da_bsl_wind = xr.DataArray(bsl_wind, dims=("lat","lon"), coords=coords)

        da_pred_members_temp = xr.DataArray(
            predictions[:, t2m_out, :, :] - 273.15, dims=("member","lat","lon"),
            coords={"member": np.arange(predictions.shape[0]), **coords}
        )
        da_pred_members_wind = xr.DataArray(
            np.hypot(predictions[:, u_out, :, :], predictions[:, v_out, :, :]),
            dims=("member","lat","lon"), coords={"member": np.arange(predictions.shape[0]), **coords}
        )

        # Compute spatial mean and assign time coordinate
        target_temp.append(
            da_tgt_temp.mean(dim=("lat","lon")).assign_coords(time=dt)
        )
        baseline_temp.append(
            da_bsl_temp.mean(dim=("lat","lon")).assign_coords(time=dt)
        )
        pred_temp.append(
            da_pred_members_temp.mean(dim=("lat","lon")).assign_coords(time=dt)
        )
        target_wind.append(
            da_tgt_wind.mean(dim=("lat","lon")).assign_coords(time=dt)
        )
        baseline_wind.append(
            da_bsl_wind.mean(dim=("lat","lon")).assign_coords(time=dt)
        )
        pred_wind.append(
            da_pred_members_wind.mean(dim=("lat","lon")).assign_coords(time=dt)
        )

        if idx % LOG_INTERVAL == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute diurnal means and stds
    temp_target_mean, _ = concat_and_group_diurnal(target_temp)
    temp_baseline_mean, _ = concat_and_group_diurnal(baseline_temp)
    temp_pred_mean, temp_pred_std = concat_and_group_diurnal(pred_temp, is_member=True)

    wind_target_mean, _ = concat_and_group_diurnal(target_wind)
    wind_baseline_mean, _ = concat_and_group_diurnal(baseline_wind)
    wind_pred_mean, wind_pred_std = concat_and_group_diurnal(pred_wind, is_member=True)

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
        temp_target_mean.hour,
        [temp_target_mean, temp_baseline_mean, temp_pred_mean],
        [None, None, temp_pred_std],
        ['COSMO-2  Analysis', 'ERA5', 'CorrDiff ± Std(Members)'],
        '2m Temperature [°C]',
        'Diurnal Cycle of 2m Temperature',
        out_root / 'diurnal_cycle_2t.png'
    )

    save_plot(
        wind_target_mean.hour,
        [wind_target_mean, wind_baseline_mean, wind_pred_mean],
        [None, None, wind_pred_std],
        ['COSMO-2  Analysis', 'ERA5', 'CorrDiff ± Std(Members)'],
        'Windspeed [m/s]',
        'Diurnal Cycle of Windspeed',
        out_root / 'diurnal_cycle_windspeed.png'
    )

    logger.info("Plots saved.")

if __name__ == '__main__':
    main()
