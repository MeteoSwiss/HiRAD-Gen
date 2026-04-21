from typing import Callable
from functools import partial
import time
from collections import defaultdict
import nvtx
import numpy as np
import random
import torch
from torch.distributed import gather
from hirad.utils.inference_utils import regression_step, diffusion_step
from hirad.distributed import DistributedManager
from hirad.utils.patching import GridPatching2D
from hirad.inference import stochastic_sampler, deterministic_sampler


def _sync_t() -> float:
    """Return wall-clock time after synchronizing all pending CUDA ops."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
class GeneratorBase():
    def __init__(self, 
                batch_size: int,
                ensemble_size: int,
                n_out_channels: int,
                dist: DistributedManager):
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.n_out_channels = n_out_channels
        self.dist = dist
        self.patching = None
        self._timings: dict[str, float] = defaultdict(float)
        self._timing_counts: dict[str, int] = defaultdict(int)
        self.get_rank_batches()

    def get_rank_batches(self, seeds=None):
        if seeds is None:
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
            self.sampler = partial(stochastic_sampler, patching=self.patching, **sampler_args)
        else:
            raise ValueError(f"Unknown sampling method {sampler_type}")

    def get_timings(self) -> dict[str, tuple[float, int]]:
        """Return accumulated timing stats: {key: (total_seconds, call_count)}."""
        return {k: (self._timings[k], self._timing_counts[k]) for k in self._timings}
class GeneratorCorrDiff(GeneratorBase):
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
        super().__init__(
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            n_out_channels=n_out_channels,
            dist=dist
        )
        self.net_reg = net_reg
        self.net_res = net_res
        self.hr_mean_conditioning = hr_mean_conditioning
        self.inference_mode = inference_mode


    def initialize_patching(self, img_shape, patch_shape, boundary_pix, overlap_pix):
        self.patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=boundary_pix,
            overlap_pix=overlap_pix,
        )

    def generate(self, image_lr, static_channels=None, date_embedding=None, lead_time_label=None, randomize=False, random_seed=None, use_apex_gn=False, skip_timing=False):
        with nvtx.annotate("generate_fn", color="green"):
            # (1, C, H, W)
            img_shape = image_lr.shape[-2:]

            _step_timings: dict = {} if not skip_timing else None
            _t = _sync_t if not skip_timing else (lambda: 0.0)

            if self.net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    _t0 = _t()
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
                        static_channels=static_channels,
                        date_embedding=date_embedding,
                        use_apex_gn=use_apex_gn,
                        _timings=_step_timings,
                    )
                    if not skip_timing:
                        self._timings["regression"] += _t() - _t0
                        self._timing_counts["regression"] += 1
            if self.net_res:
                if self.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                if randomize:
                    # Set random seed for numpy
                    if random_seed is not None:
                        np.random.seed((random_seed) % (1 << 31))
                    seeds = np.random.randint(0, 1<<31, size=self.ensemble_size)
                    self.get_rank_batches(seeds=seeds)
                with nvtx.annotate("diffusion model", color="purple"):
                    _t0 = _t()
                    image_res = diffusion_step(
                        net=self.net_res,
                        sampler_fn=self.sampler,
                        img_shape=img_shape,
                        img_out_channels=self.n_out_channels,
                        rank_batches=self.rank_batches,
                        img_lr=image_lr.expand(
                            self.batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last),
                        rank=self.dist.rank,
                        device=image_lr.device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        static_channels=static_channels,
                        date_embedding=date_embedding,
                        use_apex_gn=use_apex_gn,
                        _timings=_step_timings,
                    )
                    if not skip_timing:
                        self._timings["diffusion"] += _t() - _t0
                        self._timing_counts["diffusion"] += 1

            if not skip_timing and _step_timings:
                for k, v in _step_timings.items():
                    self._timings[k] += v
                    self._timing_counts[k] += 1
            if self.inference_mode == "regression":
                image_out = image_reg[0:1,::]
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

                _t0 = _t()
                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if self.dist.rank == 0 else None,
                    dst=0,
                )
                if not skip_timing:
                    self._timings["gather"] += _t() - _t0
                    self._timing_counts["gather"] += 1

                if self.dist.rank == 0:
                    if self.inference_mode != "regression":
                        return torch.cat(gathered_tensors), image_reg[0:1,::]
                    return torch.cat(gathered_tensors)[0:1,::], None
                else:
                    return None, None
            else:
                if self.inference_mode != "regression":
                    return image_out, image_reg[0:1,::]
                return image_out, None

class GeneratorDiT(GeneratorBase):
    def __init__(self,
                model: torch.nn.Module,
                batch_size: int,
                ensemble_size: int,
                n_out_channels: int, 
                dist: DistributedManager):
        super().__init__(
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            n_out_channels=n_out_channels,
            dist=dist
        )
        self.model = model

    def generate(self, image_lr, static_channels=None, date_embedding=None, lead_time_label=None, randomize=False, random_seed=None, use_apex_gn=True):
        with nvtx.annotate("generate_fn", color="green"):
            # (1, C, H, W)
            img_shape = image_lr.shape[-2:]

            if randomize:
                # Set random seed for numpy
                if random_seed is not None:
                    np.random.seed((random_seed) % (1 << 31))
                seeds = np.random.randint(0, 1<<31, size=self.ensemble_size)
                self.get_rank_batches(seeds=seeds)

            model_args = {}
            if date_embedding is not None:
                model_args = {"condition": date_embedding}

            with nvtx.annotate("DiT model", color="purple"):
                image_out = diffusion_step(
                    net=self.model,
                    sampler_fn=self.sampler,
                    img_shape=img_shape,
                    img_out_channels=self.n_out_channels,
                    rank_batches=self.rank_batches,
                    img_lr=image_lr.expand(
                        self.batch_size, -1, -1, -1
                    ).to(memory_format=torch.channels_last),
                    rank=self.dist.rank,
                    device=image_lr.device,
                    lead_time_label=lead_time_label,
                    static_channels=static_channels,
                    use_apex_gn=use_apex_gn,
                    additional_model_args=model_args,
                )

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
                    return torch.cat(gathered_tensors), None
                else:
                    return None, None
            else:
                return image_out, None