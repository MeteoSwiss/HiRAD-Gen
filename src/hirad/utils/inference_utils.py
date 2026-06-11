# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import os
import logging
import time

import nvtx
import numpy as np
import torch
import tqdm

from .function_utils import StackedRandomGenerator


def _sync_t() -> float:
    """Wall-clock time after flushing all pending CUDA ops on the current device."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

############################################################################
#                     CorrDiff Generation Utilities                        #
############################################################################


def regression_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    latents_shape: torch.Size,
    lead_time_label: Optional[torch.Tensor] = None,
    static_channels: Optional[torch.Tensor] = None,
    date_embedding: Optional[torch.Tensor] = None,
    use_apex_gn: bool = False,
    _timings: Optional[dict] = None,
) -> torch.Tensor:
    """
    Perform a regression step to produce ensemble mean prediction.

    This function takes a low-resolution input and performs a regression step to produce
    an ensemble mean prediction. It processes a single instance and then replicates
    the results across the batch dimension if needed.

    Parameters
    ----------
    net : torch.nn.Module
        U-Net model for regression.
    img_lr : torch.Tensor
        Low-resolution input to the network with shape (1, channels, height, width).
        Must have a batch dimension of 1.
    latents_shape : torch.Size
        Shape of the latent representation with format
        (batch_size, out_channels, image_shape_y, image_shape_x).
    lead_time_label : Optional[torch.Tensor], optional
        Lead time label tensor for lead time conditioning,
        with shape (1, lead_time_dims). Default is None.
    static_channels : torch.Tensor, optional
        Static channels input of shape (C_static, H, W).

    date_embedding : torch.Tensor, optional
        Date embedding input of shape (B, C_date).

    Returns
    -------
    torch.Tensor
        Predicted ensemble mean at the next time step with shape matching latents_shape.

    Raises
    ------
    ValueError
        If img_lr has a batch size greater than 1.
    """
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros(latents_shape, dtype=img_lr.dtype, device=img_lr.device)

    # Safety check: avoid silently ignoring batch elements in img_lr
    if img_lr.shape[0] > 1:
        raise ValueError(
            f"Expected img_lr to have a batch size of 1, "
            f"but found {img_lr.shape[0]}."
        )

    _t = _sync_t if _timings is not None else (lambda: 0.0)
    _t0 = _t()
    if static_channels is not None:
        img_lr = torch.cat(
            (img_lr, static_channels.expand(img_lr.shape[0], *static_channels.shape[1:])),
            dim=1,
        )

    if date_embedding is not None:
        date_embedding = date_embedding[:, :, None, None].expand(*date_embedding.shape[:2], *img_lr.shape[2:])
        if use_apex_gn:
            date_embedding = date_embedding.to(img_lr.dtype, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            date_embedding = date_embedding.to(img_lr.dtype, non_blocking=True).contiguous()
        img_lr = torch.cat((img_lr, date_embedding), dim=1)
    _t_prep = _t()

    # Perform regression on a single batch element
    with torch.inference_mode():
        if lead_time_label is not None:
            x = net(x=x_hat[0:1], img_lr=img_lr, lead_time_label=lead_time_label)
        else:
            x = net(x=x_hat[0:1], img_lr=img_lr, force_fp32=False)
    _t_net = _t()

    # If the batch size is greater than 1, repeat the prediction
    if x_hat.shape[0] > 1:
        x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    if _timings is not None:
        _timings["reg_input_prep"] = _timings.get("reg_input_prep", 0.0) + (_t_prep - _t0)
        _timings["reg_net_forward"] = _timings.get("reg_net_forward", 0.0) + (_t_net - _t_prep)

    return x


def diffusion_step(
    net: torch.nn.Module,
    sampler_fn: callable,
    img_shape: tuple,
    img_out_channels: int,
    rank_batches: list,
    img_lr: torch.Tensor,
    rank: int,
    device: torch.device,
    mean_hr: torch.Tensor = None,
    lead_time_label: torch.Tensor = None,
    static_channels: Optional[torch.Tensor] = None,
    date_embedding: Optional[torch.Tensor] = None,
    use_apex_gn: bool = False,
    _timings: Optional[dict] = None,
    additional_model_args: Optional[dict] = {},
    img_lr_per_sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    """
    Generate images using diffusion techniques as described in the relevant paper.

    This function applies a diffusion model to generate high-resolution images based on
    low-resolution inputs. It supports optional conditioning on high-resolution mean
    predictions and lead time labels.

    For each low-resolution sample in `img_lr`, the function generates multiple
    high-resolution samples, with different random seeds, specified in `rank_batches`.
    The function then concatenates these high-resolution samples across the batch dimension.

    Parameters
    ----------
    net : torch.nn.Module
        The diffusion model network.
    sampler_fn : callable
        Function used to sample images from the diffusion model.
    img_shape : tuple
        Shape of the images, (height, width).
    img_out_channels : int
        Number of output channels for the image.
    rank_batches : list
        List of batches of seeds to process.
    img_lr : torch.Tensor
        Low-resolution input image with shape (seed_batch_size, channels_lr, height, width).
    rank : int, optional
        Rank of the current process for distributed processing.
    device : torch.device, optional
        Device to perform computations.
    mean_hr : torch.Tensor, optional
        High-resolution mean tensor to be used as an additional input,
        with shape (1, channels_hr, height, width). Default is None.
    lead_time_label : torch.Tensor, optional
        Lead time label tensor for temporal conditioning,
        with shape (batch_size, lead_time_dims). Default is None.
    static_channels : torch.Tensor, optional
        Static channels input of shape (C_static, H, W).
    date_embedding : torch.Tensor, optional
        Date embedding input of shape (B, C_date).
    use_apex_gn : bool, optional
        Whether Apex's fused group normalization is used. Default is False.
    additional_model_args : dict, optional
        Additional arguments to pass to the model during sampling. Default is an empty dictionary.
    img_lr_per_sample : torch.Tensor, optional
        Per-sample conditioning tensor with shape
        (total_samples_this_rank, C_lr, height, width).
        When provided, each batch in ``rank_batches`` receives the appropriate
        slice of this tensor as ``img_lr`` instead of expanding the shared
        ``img_lr``.  Used for autoregressive per-member HR conditioning where
        every ensemble member has its own previous-step output as context.
        Default is None (standard shared-conditioning behaviour).

    Returns
    -------
    torch.Tensor
        Generated images concatenated across batches with shape
        (seed_batch_size * len(rank_batches), out_channels, height, width).
    """

    # Check spatial dimensions (use img_lr_per_sample when available)
    ref_shape = img_lr_per_sample.shape[-2:] if img_lr_per_sample is not None else img_lr.shape[-2:]
    if ref_shape != img_shape:
        raise ValueError(
            f"img_lr shape {ref_shape} does not match expected shape img_shape {img_shape}"
        )

    # Check mean_hr dimensions if provided
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_shape:
            raise ValueError(
                f"mean_hr shape {mean_hr.shape[2:]} does not match expected shape img_shape {img_shape}"
            )
        if mean_hr.shape[0] != 1:
            raise ValueError(f"mean_hr must have batch size 1, got {mean_hr.shape[0]}")

    if len(rank_batches) == 0:
        raise ValueError("rank_batches is empty, at least one batch of seeds is required")

    # img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if mean_hr is not None:
        additional_args["mean_hr"] = mean_hr
    if lead_time_label is not None:
        additional_args["lead_time_label"] = lead_time_label
    if static_channels is not None:
        additional_args["static_channels"] = static_channels
    if date_embedding is not None:
        additional_args["date_embedding"] = date_embedding
    additional_args["use_apex_gn"] = use_apex_gn

    _t = _sync_t if _timings is not None else (lambda: 0.0)

    # Loop over batches
    all_images = []
    _sample_offset = 0  # running index into img_lr_per_sample (when used)
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Resolve the img_lr for this batch:
            #   - img_lr_per_sample: per-member conditioning (e.g. autoregressive prev HR)
            #   - img_lr: shared conditioning broadcast to all members (existing path)
            if img_lr_per_sample is not None:
                img_lr_batch = img_lr_per_sample[_sample_offset:_sample_offset + batch_size]
                _sample_offset += batch_size
            else:
                if batch_size != img_lr.shape[0]:
                    raise ValueError(
                        f"Batch size {batch_size} does not match img_lr batch size {img_lr.shape[0]}"
                    )
                img_lr_batch = img_lr

            # Initialize random generator, and generate latents
            _t0 = _t()
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [
                    batch_size,
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ],
                device=device,
            )#.to(memory_format=torch.channels_last)
            _t_latent = _t()

            batch_timings: dict = {} if _timings is not None else None
            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr_batch, randn_like=rnd.randn_like, _timings=batch_timings, model_args=additional_model_args, **additional_args
                )
            _t_sampler = _t()

            if _timings is not None:
                _timings["diff_latent_gen"] = _timings.get("diff_latent_gen", 0.0) + (_t_latent - _t0)
                _timings["diff_sampler_total"] = _timings.get("diff_sampler_total", 0.0) + (_t_sampler - _t_latent)
                for k, v in batch_timings.items():
                    _timings[k] = _timings.get(k, 0.0) + v

            all_images.append(images)
    return torch.cat(all_images)


############################################################################
#                     Saving and Visualization Utilities                   #
############################################################################


def save_results_as_torch(output_path, time_step, image_pred, image_hr, image_lr, mean_pred):
    os.makedirs(output_path, exist_ok=True)
    if mean_pred is not None:
        torch.save(mean_pred, os.path.join(output_path, f'{time_step}-regression-prediction'))
    torch.save(image_hr, os.path.join(output_path, f'{time_step}-target'))
    torch.save(image_pred, os.path.join(output_path, f'{time_step}-predictions'))
    torch.save(image_lr, os.path.join(output_path, f'{time_step}-baseline'))


def calculate_bounds(*arrays: np.ndarray) -> tuple[float]:
    """Calculate consistent bounds across all arrays"""
    valid_arrays = [arr for arr in arrays if arr is not None]
    if not valid_arrays:
        return None, None
    
    # hanndle if there are masked arrays with invalid values (e.g. NaNs)
    all_values = []
    for arr in valid_arrays:
        if hasattr(arr, 'compressed'):  # Masked array
            compressed = arr.compressed()
            if len(compressed) > 0:
                all_values.extend(compressed)
        elif hasattr(arr, 'flatten'):  # Regular numpy array
            all_values.extend(arr.flatten())
        else:
            all_values.append(arr)
    
    if not all_values:
        return None, None
    
    vmin = min(all_values)
    vmax = max(all_values)
    return vmin, vmax
