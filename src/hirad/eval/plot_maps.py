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

            vmin, vmax = calculate_bounds(target[idx,:,:],
                                          prediction[:,idx,:,:],
                                          baseline[input_channel_idx,:,:])
            
             # set metadata 
            if channel.name == "2t":
                err_vmin, err_vmax = 0, 4.5
                err_colormap = "RdBu"
                unit = "K"
            elif channel.name == "tp":
                err_vmin, err_vmax = 0, 10
                err_colormap = "blues"
                unit = "mm/h"
            elif channel.name == "u10" or channel.name == "v10":
                err_vmin, err_vmax = -10, 10
                err_colormap = "BrBG"
                unit = "m/s"
            else:
                err_vmin, err_vmax = vmin, vmax

            # Prepare title and label
            plot_title = f"{getattr(channel, 'title', channel.name)}"


            # Plot target
            plot_map(
                target[idx, :, :], latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-target'),
                vmin=vmin, vmax=vmax,
                title=plot_title,
                label=unit
            )

            # Plot baseline
            plot_map(
                baseline[input_channel_idx, :, :], latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline'),
                vmin=vmin, vmax=vmax,
                title=plot_title,
                label=unit
            )

            # Plot baseline MAE
            _, baseline_mae = compute_mae(baseline[input_channel_idx, :, :], target[idx, :, :])

            plot_map(
                baseline_mae.reshape(baseline[input_channel_idx, :, :].shape), latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline-mae'),
                vmin=err_vmin, vmax=err_vmax,
                title=f"{plot_title} MAE",
                label=unit
            )

            # Plot baseline mean error (difference)
            baseline_me = (baseline[input_channel_idx, :, :] - target[idx, :, :])
            baseline_me_cmap = err_colormap if channel.name == "2t" else None
            plot_map(
                baseline_me, latitudes, longitudes,
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline-me'),
                vmin=-err_vmax, vmax=err_vmax,
                cmap=baseline_me_cmap,
                title=f"{plot_title} Mean Error",
                label=unit
            )

            if prediction.shape[0] > 1:
                for member_idx in range(prediction.shape[0]):
                    plot_map(
                        prediction[member_idx,idx,:,:], latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}'), 
                        vmin=vmin, vmax=vmax,
                        title=plot_title,
                        label=unit
                    )
                    # Plot prediction MAE for each ensemble member
                    _, prediction_mae = compute_mae(prediction[member_idx,idx,:,:], target[idx, :, :])
                    plot_map(
                        prediction_mae.reshape(prediction[member_idx,idx,:,:].shape), latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}-mae'),
                        vmin=err_vmin, vmax=err_vmax,
                        title=f"{plot_title} MAE",
                        label=unit
                    )
                    # Plot prediction mean error for each ensemble member
                    prediction_me = (prediction[member_idx,idx,:,:] - target[idx, :, :])
                    prediction_me_cmap = err_colormap if channel.name == "2t" else None
                    plot_map(
                        prediction_me, latitudes, longitudes,
                        os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}-me'),
                        vmin=-err_vmax, vmax=err_vmax,
                        cmap=prediction_me_cmap,
                        title=f"{plot_title} Mean Error",
                        label=unit
                    )
            else:
                plot_map(
                    prediction[0,idx,:,:], latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction'), 
                    vmin=vmin, vmax=vmax,
                    title=plot_title,
                    label=unit
                )
                # Plot prediction MAE for single prediction
                _, prediction_mae = compute_mae(prediction[0,idx,:,:], target[idx, :, :])
                plot_map(
                    prediction_mae, latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction-mae'),
                    vmin=err_vmin, vmax=err_vmax,
                    title=f"{plot_title} MAE",
                    label=unit
                )
                # Plot prediction mean error for single prediction
                prediction_me = (prediction[0,idx,:,:] - target[idx, :, :])
                prediction_me_cmap = err_colormap if channel.name == "2t" else None
                plot_map(
                    prediction_me, latitudes, longitudes,
                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction-me'),
                    vmin=-err_vmax, vmax=err_vmax,
                    cmap=prediction_me_cmap,
                    title=f"{plot_title} Mean Error",
                    label=unit
                )
                    
    logger.info("Image loading and plotting completed.")

if __name__ == "__main__":
    main()
