import hydra
import os
import logging
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
import sys

from hirad.distributed import DistributedManager
from hirad.utils.inference_utils import calculate_bounds, _prepare_precipitation
from hirad.utils.function_utils import get_time_from_range
from hirad.eval import compute_mae, plot_map

from hirad.datasets import get_dataset_and_sampler_inference

def load_data(output_path, time=None, filename=None):
    return torch.load(os.path.join(output_path, time, filename), weights_only=False)

def map_output_to_input_channels(output_channels, input_channels):
    """
    Maps output channels to input channels based on their names.
    """
    return {
        j: next((k for k, input_channel in enumerate(input_channels) if input_channel.name == output_channel.name), -1)
        for j, output_channel in enumerate(output_channels)
    }

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:

    # Set distributed environment variables
    if "MASTER_ADDR" not in os.environ:
        master_node = os.environ["SLURM_JOB_NODELIST"].split()[0]
        master_addr = os.popen(f"getent ahosts {master_node} | awk '{{ print $1; exit }}'").read().strip()
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("plot_maps")

    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") 
        
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler_inference(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )

    output_path = getattr(cfg.generation.io, "output_path", "./outputs")

    for curr_time in times:
        prediction = load_data(output_path, time=curr_time, filename=f'{curr_time}-predictions')
        baseline = load_data(output_path, time=curr_time, filename=f'{curr_time}-baseline')
        target = load_data(output_path, time=curr_time, filename=f'{curr_time}-target')

        # Compute output_channels as in compute_eval.py
        input_channels = dataset.input_channels()
        output_channels = dataset.output_channels()

        longitudes = dataset.longitude()
        latitudes = dataset.latitude()

        # Map output channels to input channels
        output_to_input_channel_map = map_output_to_input_channels(output_channels, input_channels)

        for idx, channel in enumerate(output_channels):
            channel_dir = channel.name + "_" + channel.level if channel.level else channel.name
            output_path_channel = os.path.join(output_path, channel_dir)
            os.makedirs(output_path_channel, exist_ok=True)

            input_channel_idx = output_to_input_channel_map[idx]

            # Specialized handling for precipitation data
            if channel.name == "tp":
                target[idx, :, :] = _prepare_precipitation(target[idx, :, :])
                prediction[:, idx, :, :] = _prepare_precipitation(prediction[:, idx, :, :])
                baseline[input_channel_idx, :, :] = _prepare_precipitation(baseline[input_channel_idx, :, :])

            cmap = "viridis"
            vmin, vmax = calculate_bounds(target[idx,:,:],
                                          prediction[:,idx,:,:],
                                          baseline[input_channel_idx,:,:])
            
             # set metadata 
            if channel.name == "2t":
                err_vmin, err_vmax = -4.5, 4.5 # err_vmin hard coded to 0 for mae
                cmap = "RdYlBu_r"
                me_cmap = "RdBu"
                unit = "K"
            elif channel.name == "tp":
                err_vmin, err_vmax = -10, 10 # Remove?
                me_cmap = "blues"
                unit = "mm/h"
            elif channel.name == "10u" or channel.name == "10v":
                vmin, vmax = -10, 10
                err_vmin, err_vmax = vmin, vmax # err_vmin hard coded to 0 for mae
                cmap = "BrBG"
                me_cmap = cmap
                unit = "m/s"
            else:
                err_vmin, err_vmax = vmin, vmax

            # Reformat curr_time for plot title
            dt_str = curr_time
            from datetime import datetime
            dt = datetime.strptime(dt_str, "%Y%m%d-%H%M")
            formatted_time = dt.strftime("%d-%m-%Y %H:%M")

            # Prepare title and label
            plot_title = f"{formatted_time}: {getattr(channel, 'title', channel.name)}"


            # Plot target
            plot_map(
                target[idx, :, :], latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-target'),
                vmin=vmin, vmax=vmax,
                title=plot_title,
                label=unit,
                extend='max' if channel.name == "tp" else 'both',
                cmap=cmap
            )

            # Plot baseline
            plot_map(
                baseline[input_channel_idx, :, :], latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline'),
                vmin=vmin, vmax=vmax,
                title=plot_title,
                label=unit,
                extend='max' if channel.name == "tp" else 'both',
                cmap=cmap
            )

            # Plot baseline MAE
            _, baseline_mae = compute_mae(baseline[input_channel_idx, :, :], target[idx, :, :])

            if channel.name != "tp":
                plot_map(
                    baseline_mae.reshape(baseline[input_channel_idx, :, :].shape), latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline-mae'),
                    vmin=0, vmax=err_vmax,
                    title=f"{plot_title} MAE",
                    label=unit,
                    extend='max',
                    cmap=cmap if channel.name not in ("10u", "10v", "2t") else 'viridis'
                )

            # Plot baseline mean error (difference)
            baseline_me = (baseline[input_channel_idx, :, :] - target[idx, :, :])
            baseline_me_cmap = me_cmap if channel.name == "2t" else None

            if channel.name != "tp":
                plot_map(
                    baseline_me, latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline-me'),
                    vmin=err_vmin, vmax=err_vmax,
                    cmap=me_cmap,
                    title=f"{plot_title} Mean Error",
                    label=unit,
                    extend='both'
                )

            if prediction.shape[0] > 1:
                for member_idx in range(prediction.shape[0]):
                    plot_map(
                        prediction[member_idx,idx,:,:], latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}'), 
                        vmin=vmin, vmax=vmax,
                        title=plot_title,
                        label=unit,
                        extend='max' if channel.name == "tp" else 'both',
                        cmap=cmap
                    )
                    # Plot prediction MAE for each ensemble member
                    _, prediction_mae = compute_mae(prediction[member_idx,idx,:,:], target[idx, :, :])
                    if channel.name != "tp":
                        plot_map(
                            prediction_mae.reshape(prediction[member_idx,idx,:,:].shape), latitudes, longitudes,
                            os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}-mae'),
                            vmin=0, vmax=err_vmax,
                            title=f"{plot_title} MAE",
                            label=unit,
                            extend='max',
                            cmap=cmap if channel.name not in ("10u", "10v", "2t") else 'viridis'
                        )
                    # Plot prediction mean error for each ensemble member
                    prediction_me = (prediction[member_idx,idx,:,:] - target[idx, :, :])
                    prediction_me_cmap = me_cmap if channel.name == "2t" else None

                    if channel.name != "tp":
                        plot_map(
                            prediction_me, latitudes, longitudes,
                            os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}-me'),
                            vmin=err_vmin, vmax=err_vmax,
                            cmap=me_cmap,
                            title=f"{plot_title} Mean Error",
                            label=unit,
                            extend='both'
                        )
            else:
                plot_map(
                    prediction[0,idx,:,:], latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction'), 
                    vmin=vmin, vmax=vmax,
                    title=plot_title,
                    label=unit,
                    extend='max' if channel.name == "tp" else 'both',
                    cmap=cmap
                )
                # Plot prediction MAE for single prediction
                _, prediction_mae = compute_mae(prediction[0,idx,:,:], target[idx, :, :])
                if channel.name != "tp":
                    plot_map(
                        prediction_mae, latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction-mae'),
                        vmin=0, vmax=err_vmax,
                        title=f"{plot_title} MAE",
                        label=unit,
                        extend='max',
                        cmap=cmap if channel.name not in ("10u", "10v", "2t") else 'viridis'
                    )
                # Plot prediction mean error for single prediction
                prediction_me = (prediction[0,idx,:,:] - target[idx, :, :])
                prediction_me_cmap = me_cmap if channel.name == "2t" else None

                if channel.name != "tp":
                    plot_map(
                        prediction_me, latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction-me'),
                        vmin=err_vmin, vmax=err_vmax,
                        cmap=me_cmap,
                        title=f"{plot_title} Mean Error",
                        label=unit,
                        extend='both'
                    )

        # Plot Windspeed and direction
        # Find indices for 10u and 10v channels
        wind_channels = {ch.name: idx for idx, ch in enumerate(output_channels) if ch.name in ("10u", "10v")}
        if "10u" in wind_channels and "10v" in wind_channels:
            idx_10u = wind_channels["10u"]
            idx_10v = wind_channels["10v"]
            input_idx_10u = output_to_input_channel_map[idx_10u]
            input_idx_10v = output_to_input_channel_map[idx_10v]

            # Compute windspeed and direction for target, baseline, prediction
            def compute_wind(u, v):
                speed = np.sqrt(u**2 + v**2)
                direction = (np.arctan2(-u, -v) * 180 / np.pi) % 360
                return speed, direction

            target_wind_speed, target_wind_dir = compute_wind(target[idx_10u, :, :], target[idx_10v, :, :])
            baseline_wind_speed, baseline_wind_dir = compute_wind(baseline[input_idx_10u, :, :], baseline[input_idx_10v, :, :])
            prediction_wind_speed = []
            prediction_wind_dir = []
            for member_idx in range(prediction.shape[0]):
                ws, wd = compute_wind(prediction[member_idx, idx_10u, :, :], prediction[member_idx, idx_10v, :, :])
                prediction_wind_speed.append(ws)
                prediction_wind_dir.append(wd)
            prediction_wind_speed = np.stack(prediction_wind_speed)
            prediction_wind_dir = np.stack(prediction_wind_dir)

            # Reformat curr_time for plot title
            dt_str = curr_time
            from datetime import datetime
            dt = datetime.strptime(dt_str, "%Y%m%d-%H%M")
            formatted_time = dt.strftime("%d-%m-%Y %H:%M")

            # Plot windspeed
            wind_cmap = "viridis"
            wind_vmin, wind_vmax = 0, 10
            wind_unit = "m/s"
            plot_title_speed = f"{formatted_time}: FF10m"
            os.makedirs(os.path.join(output_path, "FF10m"), exist_ok=True)
            plot_map(
                target_wind_speed, latitudes, longitudes,
                os.path.join(output_path, "FF10m", f'{curr_time}-FF10m-target'),
                vmin=wind_vmin, vmax=wind_vmax,
                title=plot_title_speed,
                label=wind_unit,
                extend='max',
                cmap=wind_cmap
            )
            plot_map(
                baseline_wind_speed, latitudes, longitudes,
                os.path.join(output_path, "FF10m", f'{curr_time}-FF10m-baseline'),
                vmin=wind_vmin, vmax=wind_vmax,
                title=plot_title_speed,
                label=wind_unit,
                extend='max',
                cmap=wind_cmap
            )
            for member_idx in range(prediction.shape[0]):
                plot_map(
                    prediction_wind_speed[member_idx], latitudes, longitudes,
                    os.path.join(output_path, "FF10m", f'{curr_time}-FF10m-prediction_{member_idx}'),
                    vmin=wind_vmin, vmax=wind_vmax,
                    title=plot_title_speed,
                    label=wind_unit,
                    extend='max',
                    cmap=wind_cmap
                )

            # Plot wind direction
            dir_cmap = "twilight"
            dir_vmin, dir_vmax = 0, 360
            dir_unit = "deg"
            plot_title_dir = f"{formatted_time}: DD10m"
            os.makedirs(os.path.join(output_path, "DD10m"), exist_ok=True)
            plot_map(
                target_wind_dir, latitudes, longitudes,
                os.path.join(output_path, "DD10m", f'{curr_time}-DD10m-target'),
                vmin=dir_vmin, vmax=dir_vmax,
                title=plot_title_dir,
                label=dir_unit,
                extend='both',
                cmap=dir_cmap
            )
            plot_map(
                baseline_wind_dir, latitudes, longitudes,
                os.path.join(output_path, "DD10m", f'{curr_time}-DD10m-baseline'),
                vmin=dir_vmin, vmax=dir_vmax,
                title=plot_title_dir,
                label=dir_unit,
                extend='both',
                cmap=dir_cmap
            )
            for member_idx in range(prediction.shape[0]):
                plot_map(
                    prediction_wind_dir[member_idx], latitudes, longitudes,
                    os.path.join(output_path, "DD10m", f'{curr_time}-DD10m-prediction_{member_idx}'),
                    vmin=dir_vmin, vmax=dir_vmax,
                    title=plot_title_dir,
                    label=dir_unit,
                    extend='both',
                    cmap=dir_cmap
                )


    logger.info("Image loading and plotting completed.")

if __name__ == "__main__":
    main()
