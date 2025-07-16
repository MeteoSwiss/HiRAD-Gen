import hydra
import os
import json
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import numpy as np
import contextlib

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from concurrent.futures import ThreadPoolExecutor

from hirad.eval import compute_crps_over_time, plot_crps_over_time_and_area, compute_crps_over_time_and_area
from hirad.models import EDMPrecondSuperResolution, UNet
from hirad.inference import Generator
from hirad.utils.inference_utils import save_images, save_results_as_torch
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None: 

    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    
    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)


    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") 
    
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler_inference(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    output_path = getattr(cfg.generation.io, "output_path", "./outputs")

    #plot_crps_over_time_and_area(times, dataset, output_path)
    #compute_crps_over_time(times, dataset, output_path)
    compute_crps_over_time_and_area(times, dataset, output_path)
    



if __name__ == "__main__":
    main()