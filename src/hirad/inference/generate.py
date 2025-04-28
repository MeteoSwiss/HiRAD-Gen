import hydra
import os
import json
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import nvtx
import numpy as np
from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from einops import rearrange
from torch.distributed import gather


from hydra.utils import to_absolute_path
from hirad.models import EDMPrecond, UNet
from hirad.utils.stochastic_sampler import stochastic_sampler
from hirad.utils.deterministic_sampler import deterministic_sampler
from hirad.utils.inference_utils import (
    get_time_from_range,
    regression_step,
    diffusion_step,
)
from hirad.utils.checkpoint import load_checkpoint


from hirad.utils.generate_utils import (
    get_dataset_and_sampler
)

from hirad.utils.train_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random dowscaled atmospheric states using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

        # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse the inference input times
    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range) #TODO check what time formats we are using and adapt
    else:
        times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Parse the patch shape
    if hasattr(cfg.generation, "patch_shape_x"):  # TODO better config handling
        patch_shape_x = cfg.generation.patch_shape_x
    else:
        patch_shape_x = None
    if hasattr(cfg.generation, "patch_shape_y"):
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_path = cfg.generation.io.res_ckpt_path
        logger0.info(f'Loading residual network from "{res_ckpt_path}"...')

        diffusion_model_args_path = os.path.join(res_ckpt_path, 'model_args.json')
        if not os.path.isfile(diffusion_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{diffusion_model_args_path}'.")
        with open(diffusion_model_args_path, 'r') as f:
            diffusion_model_args = json.load(f)

        net_res = EDMPrecond(**diffusion_model_args)

        _ = load_checkpoint(
            path=res_ckpt_path,
            model=net_res,
            device=dist.device
        )
        
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_path = cfg.generation.io.reg_ckpt_path
        logger0.info(f'Loading network from "{reg_ckpt_path}"...')


        regression_model_args_path = os.path.join(reg_ckpt_path, 'model_args.json')
        if not os.path.isfile(regression_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{regression_model_args_path}'.")
        with open(regression_model_args_path, 'r') as f:
            regression_model_args = json.load(f)

        net_reg = EDMPrecond(**regression_model_args)

        _ = load_checkpoint(
            path=reg_ckpt_path,
            model=net_reg,
            device=dist.device
        )
        
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
    else:
        net_reg = None