"""Thin wrappers around manual inference model-loading utilities."""

from __future__ import annotations

import json

import torch

from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _resolve_device,
)

from .types import PredictionConfig


def setup_distributed(config: PredictionConfig) -> tuple[str, object | None, int, int, int]:
    """Resolve device and initialize the model communication group."""

    global_rank, local_rank, world_size = _get_parallel_info()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but CUDA is not available on this host. "
            "Refusing to fall back to CPU for diffusion sampling."
        )

    device = _resolve_device(config.device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    if config.num_gpus_per_model > 1 and world_size != config.num_gpus_per_model:
        raise SystemExit(
            f"Expected world_size={config.num_gpus_per_model} for model-parallel inference, "
            f"got {world_size}. Launch with matching srun --ntasks."
        )

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)
    return device, model_comm_group, global_rank, local_rank, world_size


def load_inference_model(
    config: PredictionConfig,
) -> tuple[object, object, dict, str, object | None, int, int, int]:
    """Load the inference model, datamodule, parsed sampling args, and runtime metadata."""

    device, model_comm_group, global_rank, local_rank, world_size = setup_distributed(config)
    extra_args = json.loads(config.extra_args_json) if config.extra_args_json else {}
    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=str(config.checkpoint_path),
        device=device,
        validation_frequency=config.validation_frequency,
        precision=config.precision,
        num_gpus_per_model_override=config.num_gpus_per_model,
    )
    return (
        inference_model,
        datamodule,
        extra_args,
        device,
        model_comm_group,
        global_rank,
        local_rank,
        world_size,
    )
