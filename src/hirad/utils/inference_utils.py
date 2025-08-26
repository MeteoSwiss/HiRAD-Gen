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

import nvtx
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from .function_utils import StackedRandomGenerator
from hirad.eval import compute_mae, average_power_spectrum, plot_error_projection, plot_power_spectra, crps

############################################################################
#                     CorrDiff Generation Utilities                        #
############################################################################


def regression_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    latents_shape: torch.Size,
    lead_time_label: Optional[torch.Tensor] = None,
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

    # Perform regression on a single batch element
    with torch.inference_mode():
        if lead_time_label is not None:
            x = net(x=x_hat[0:1], img_lr=img_lr, lead_time_label=lead_time_label)
        else:
            x = net(x=x_hat[0:1], img_lr=img_lr, force_fp32=False)

    # If the batch size is greater than 1, repeat the prediction
    if x_hat.shape[0] > 1:
        x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

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

    Returns
    -------
    torch.Tensor
        Generated images concatenated across batches with shape
        (seed_batch_size * len(rank_batches), out_channels, height, width).
    """

    # Check img_lr dimensions match expected shape
    if img_lr.shape[2:] != img_shape:
        raise ValueError(
            f"img_lr shape {img_lr.shape[2:]} does not match expected shape img_shape {img_shape}"
        )

    # Check mean_hr dimensions if provided
    if mean_hr is not None:
        if mean_hr.shape[2:] != img_shape:
            raise ValueError(
                f"mean_hr shape {mean_hr.shape[2:]} does not match expected shape img_shape {img_shape}"
            )
        if mean_hr.shape[0] != 1:
            raise ValueError(f"mean_hr must have batch size 1, got {mean_hr.shape[0]}")

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if mean_hr is not None:
        additional_args["mean_hr"] = mean_hr
    if lead_time_label is not None:
        additional_args["lead_time_label"] = lead_time_label

    # Loop over batches
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [
                    img_lr.shape[0],
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ],
                device=device,
            )#.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
                )
            all_images.append(images)
    return torch.cat(all_images)


############################################################################
#                           Visualization Utilities                        #
############################################################################


def save_results_as_torch(output_path, time_step, dataset, image_pred, image_hr, image_lr, mean_pred):
    
    os.makedirs(output_path, exist_ok=True)
    target = np.flip(dataset.denormalize_output(image_hr[0,::].squeeze()),1)
    prediction_ensemble = np.flip(dataset.denormalize_output(image_pred.squeeze()),-2)
    baseline = np.flip(dataset.denormalize_input(image_lr[0,::].squeeze()),1)
    if mean_pred is not None:
        mean_pred = np.flip(dataset.denormalize_output(mean_pred[0,::].squeeze()),1)
        torch.save(mean_pred, os.path.join(output_path, f'{time_step}-regression-prediction'))
    torch.save(target, os.path.join(output_path, f'{time_step}-target'))
    torch.save(prediction_ensemble, os.path.join(output_path, f'{time_step}-predictions'))
    torch.save(baseline, os.path.join(output_path, f'{time_step}-baseline'))

@DeprecationWarning
def save_images(output_path, time_step, dataset, image_pred, image_hr, image_lr, mean_pred):   

    os.makedirs(output_path, exist_ok=True)

    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()

    target = np.flip(dataset.denormalize_output(image_hr[0,::].squeeze()),1) #.reshape(len(output_channels),-1)
    prediction = np.flip(dataset.denormalize_output(image_pred),-2) #.reshape(len(output_channels),-1)
    baseline = np.flip(dataset.denormalize_input(image_lr[0,::].squeeze()),1)# .reshape(len(input_channels),-1) 
    if mean_pred is not None:
        mean_pred = np.flip(dataset.denormalize_output(mean_pred[0,::].squeeze()),1) #.reshape(len(output_channels),-1)

    #  Plot power spectra
    freqs = {}
    power = {}
    for idx, channel in enumerate(output_channels):
        channel_dir = channel.name + "_" + channel.level if channel.level else channel.name
        output_path_channel = os.path.join(output_path, channel_dir)
        if not os.path.exists(output_path_channel):
            os.makedirs(output_path_channel)
        input_channel_idx = input_channels.index(channel)

        if channel.name=="tp":
            target[idx,::] = transform_channel(target[idx,:,:])
            prediction[:,idx,::] = transform_channel(prediction[:,idx,:,:])
            baseline[input_channel_idx,:,:] = transform_channel(baseline[input_channel_idx,::])
            if mean_pred is not None:
                mean_pred[idx,::] = transform_channel(mean_pred[idx,::])
        
        if mean_pred is not None:
            vmin, vmax = calculate_bounds(target[idx,:,:],
                                            prediction[:,idx,:,:],
                                            baseline[input_channel_idx,:,:],
                                            mean_pred[idx,:,:])
        else:
            vmin, vmax = calculate_bounds(target[idx,:,:],
                                            prediction[:,idx,:,:],
                                            baseline[input_channel_idx,:,:])      
        _plot_projection(longitudes, latitudes, target[idx,:,:],
                          os.path.join(output_path_channel, f'{time_step}-{channel.name}-target.jpg'), 
                          vmin=vmin, vmax=vmax)
        if prediction.shape[0] > 1:
            for member_idx in range(prediction.shape[0]):
                _plot_projection(longitudes, latitudes, prediction[member_idx,idx,:,:],
                                  os.path.join(output_path_channel, f'{time_step}-{channel.name}-prediction_{member_idx}.jpg'), 
                                  vmin=vmin, vmax=vmax)
        else:
            _plot_projection(longitudes, latitudes, 
                             prediction[0,idx,:,:], os.path.join(output_path_channel, f'{time_step}-{channel.name}-prediction.jpg'), 
                             vmin=vmin, vmax=vmax)
        _plot_projection(longitudes, latitudes, baseline[input_channel_idx,:,:], 
                         os.path.join(output_path_channel, f'{time_step}-{channel.name}-input.jpg'), 
                         vmin=vmin, vmax=vmax)
        if mean_pred is not None:
            _plot_projection(longitudes, latitudes, mean_pred[idx,:,:], 
                             os.path.join(output_path_channel, f'{time_step}-{channel.name}-mean_prediction.jpg'), 
                             vmin=vmin, vmax=vmax)

        _, baseline_errors = compute_mae(baseline[input_channel_idx,:,:], target[idx,:,:])
        plot_error_projection(baseline_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path_channel, f'{time_step}-{channel.name}-baseline-error.jpg'))
        if prediction.shape[0] > 1:
            for member_idx in range(prediction.shape[0]):
                _, prediction_errors = compute_mae(prediction[member_idx,idx,:,:], target[idx,:,:])
                plot_error_projection(prediction_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path_channel, f'{time_step}-{channel.name}-prediction_{member_idx}-error.jpg'))
        else:
            _, prediction_errors = compute_mae(prediction[0,idx,:,:], target[idx,:,:])
            plot_error_projection(prediction_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path_channel, f'{time_step}-{channel.name}-prediction-error.jpg'))            
        if mean_pred is not None:
            _, mean_prediction_errors = compute_mae(mean_pred[idx,:,:], target[idx,:,:])
            plot_error_projection(mean_prediction_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path_channel, f'{time_step}-{channel.name}-mean-prediction-error.jpg'))

        b_freq, b_power = average_power_spectrum(baseline[input_channel_idx,:,:].squeeze(), 2.0)
        freqs['baseline'] = b_freq
        power['baseline'] = b_power
        #plotting.plot_power_spectrum(b_freq, b_power, target_channels[t_c], os.path.join('plots/spectra/baseline2dt',  target_channels[t_c] + '-all_dates'))
        t_freq, t_power = average_power_spectrum(target[idx,:,:].squeeze(), 2.0)
        freqs['target'] = t_freq
        power['target'] = t_power
        p_freq, p_power = average_power_spectrum(prediction[-1,idx,:,:].squeeze(), 2.0)
        freqs['prediction'] = p_freq
        power['prediction'] = p_power
        if mean_pred is not None:
            mp_freq, mp_power = average_power_spectrum(mean_pred[idx,:,:].squeeze(), 2.0)
            freqs['mean_prediction'] = mp_freq
            power['mean_prediction'] = mp_power
        plot_power_spectra(freqs, power, channel.name, os.path.join(output_path_channel, f'{time_step}-{channel.name}-spectra.jpg'))

def transform_channel(channel_array, channel_name="tp"):
    # precip_array = np.clip(precip_array, 0, None)
    # precip_array = np.where(precip_array == 0, 1e-6, precip_array)
    # epsilon = 1e-2
    # precip_array = precip_array + epsilon
    # precip_array = np.log10(precip_array)
    # log_min, log_max = precip_array.min(), precip_array.max()
    # precip_array = (precip_array-log_min)/(log_max-log_min)
    if channel_name == "tp":
        channel_array = np.clip(channel_array, 0, None)
        channel_array = (np.power(channel_array,0.25)-1)/0.25
    elif channel_name == "2t":
        channel_array = channel_array - 273.15
    # precip_array = np.sqrt(precip_array)
    return channel_array

@DeprecationWarning
def _plot_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str, cmap=None, vmin = None, vmax = None):

    """Plot observed or interpolated data in a scatter plot."""
    # TODO: Refactor this somehow, it's not really generalizing well across variables.
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, orientation="horizontal")
    plt.savefig(filename)
    plt.close('all')

def calculate_bounds(*arrays: np.ndarray) -> tuple[float]:
    """Calculate consistent bounds across all arrays"""
    valid_arrays = [arr for arr in arrays if arr is not None]
    if not valid_arrays:
        return 0, 1
    
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
        return 0, 1
    
    vmin = min(all_values)
    vmax = max(all_values)
    return vmin, vmax