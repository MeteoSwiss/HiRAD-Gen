import hydra
import os
import logging
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig
import sys

from hirad.distributed import DistributedManager
from hirad.utils.inference_utils import save_images, _plot_projection, calculate_bounds
from hirad.utils.function_utils import get_time_from_range

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
            vmin, vmax = calculate_bounds(target[idx,:,:],
                                          prediction[:,idx,:,:],
                                          baseline[input_channel_idx,:,:])

            # Plot target
            _plot_projection(
                longitudes, latitudes, target[idx, :, :],
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-target.jpg'),
                vmin=vmin, vmax=vmax
            )

            # Plot baseline
            _plot_projection(
                longitudes, latitudes, baseline[input_channel_idx, :, :],
                os.path.join(output_path_channel, f'{curr_time}-{channel.name}-baseline.jpg'),
                vmin=vmin, vmax=vmax
            )

            if prediction.shape[0] > 1:
                for member_idx in range(prediction.shape[0]):
                    _plot_projection(longitudes, latitudes, prediction[member_idx,idx,:,:],
                                    os.path.join(output_path_channel, f'{curr_time}-{channel.name}-prediction_{member_idx}.jpg'), 
                                    vmin=vmin, vmax=vmax)
                    
    logger.info("Image loading and plotting completed.")

if __name__ == "__main__":
    main()
