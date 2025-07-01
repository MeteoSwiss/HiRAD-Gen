from typing import Callable
from functools import partial
import nvtx
import numpy as np
import torch
from torch.distributed import gather
from hirad.utils.inference_utils import regression_step, diffusion_step
from hirad.distributed import DistributedManager
from hirad.utils.patching import GridPatching2D
from hirad.inference import stochastic_sampler, deterministic_sampler

class Generator():
    def __init__(self, 
                net_reg: torch.nn.Module, 
                net_res: torch.nn.Module,
                batch_size: int,
                ensemble_size: int,
                hr_mean_conditioning: bool, 
                n_out_channels: int, 
                inference_mode: str,
                dist: DistributedManager,
                ):
        
        self.net_reg = net_reg
        self.net_res = net_res
        self.batch_size = batch_size
        self.hr_mean_conditioning = hr_mean_conditioning
        self.n_out_channels = n_out_channels
        self.inference_mode = inference_mode
        self.ensemble_size = ensemble_size
        self.dist = dist
        self.get_rank_batches()
        self.patching = None

    def get_rank_batches(self):
        seeds = list(np.arange(self.ensemble_size))
        num_batches = (
            (len(seeds) - 1) // (self.batch_size * self.dist.world_size) + 1
        ) * self.dist.world_size
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        self.rank_batches = all_batches[self.dist.rank :: self.dist.world_size]

    def initialize_sampler(self, sampler_type, **sampler_args):
        if sampler_type == "deterministic":
            if self.hr_mean_conditioning:
                raise NotImplementedError(
                    "High-res mean conditioning is not yet implemented for the deterministic sampler"
                )
            self.sampler = partial(
                deterministic_sampler,
                **sampler_args
            )
        elif sampler_type == "stochastic":
            self.sampler = partial(stochastic_sampler, patching=self.patching)
        else:
            raise ValueError(f"Unknown sampling method {sampler_type}")

    def initialize_patching(self, img_shape, patch_shape, boundary_pix, overlap_pix):
        self.patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=boundary_pix,
            overlap_pix=overlap_pix,
        )

    def generate(self, image_lr, lead_time_label=None):
        with nvtx.annotate("generate_fn", color="green"):
            # (1, C, H, W)
            image_lr = image_lr.to(memory_format=torch.channels_last)
            img_shape = image_lr.shape[-2:]

            if self.net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=self.net_reg,
                        img_lr=image_lr,
                        latents_shape=(
                            self.batch_size,
                            self.n_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ), # (batch_size, C, H, W)
                        lead_time_label=lead_time_label,
                    )
            if self.net_res:
                if self.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=self.net_res,
                        sampler_fn=self.sampler,
                        img_shape=img_shape,
                        img_out_channels=self.n_out_channels,
                        rank_batches=self.rank_batches,
                        img_lr=image_lr.expand(
                            self.batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last), #.to(memory_format=torch.channels_last),
                        rank=self.dist.rank,
                        device=image_lr.device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                    )
            if self.inference_mode == "regression":
                image_out = image_reg
            elif self.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg[0:1,::] + image_res

            # Gather tensors on rank 0
            if self.dist.world_size > 1:
                if self.dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(self.dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if self.dist.rank == 0 else None,
                    dst=0,
                )

                if self.dist.rank == 0:
                    if self.inference_mode != "regression":
                        return torch.cat(gathered_tensors), image_reg[0:1,::]
                    return torch.cat(gathered_tensors), None
                else:
                    return None, None
            else:
                #TODO do this for multi-gpu setting above too
                if self.inference_mode != "regression":
                    return image_out, image_reg
                return image_out, None
