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

import eccodes
import nvtx
import numpy as np
import torch
import tqdm
import earthkit.data as ekd
from earthkit.data import FieldList

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

    Returns
    -------
    torch.Tensor
        Generated images concatenated across batches with shape
        (seed_batch_size * len(rank_batches), out_channels, height, width).
    """

    # Check img_lr dimensions match expected shape
    if img_lr.shape[-2:] != img_shape:
        raise ValueError(
            f"img_lr shape {img_lr.shape[-2:]} does not match expected shape img_shape {img_shape}"
        )

    # Check mean_hr dimensions if provided
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_shape:
            raise ValueError(
                f"mean_hr shape {mean_hr.shape[-2:]} does not match expected shape img_shape {img_shape}"
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
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue
            if batch_size != img_lr.shape[0]:
                raise ValueError(
                    f"Batch size {batch_size} does not match img_lr batch size {img_lr.shape[0]}"
                )

            # Initialize random generator, and generate latents
            _t0 = _t()
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
            _t_latent = _t()

            batch_timings: dict = {} if _timings is not None else None
            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr, randn_like=rnd.randn_like,
                    _timings=batch_timings, **additional_args
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


def save_results(output_path, time_step, dataset, image_pred, image_hr, image_lr, mean_pred, base_time=None, output_format='torch', grib_template_path=''):
    if base_time is not None:
        # base_time is only defined for forecast-type datasets (AnemoiForecastDataset);
        # nest under it to keep overlapping (reference_time, step) pairs from colliding
        # on valid_time alone.
        savedir = os.path.join(output_path, base_time, time_step)
    else:
        savedir = os.path.join(output_path, time_step)
    os.makedirs(savedir, exist_ok=True)
    # Data arrives already denormalized and spatially oriented (physical units, numpy)
    target = image_hr
    prediction_ensemble = image_pred
    baseline = image_lr
    if output_format == 'torch':
        save_results_as_torch(savedir, time_step, target, prediction_ensemble, baseline, mean_pred)
    elif output_format == 'grib':
        save_results_as_grib(savedir, time_step, target, prediction_ensemble, baseline, mean_pred, dataset, grib_template_path)
    elif output_format == 'both':
        save_results_as_torch(savedir, time_step, target, prediction_ensemble, baseline, mean_pred)
        save_results_as_grib(savedir, time_step, target, prediction_ensemble, baseline, mean_pred, dataset, grib_template_path)
    else:
        raise ValueError(f'output format {output_format} not supported-- torch or grib or both supported')

def save_results_as_torch(output_path, time_step, target, prediction_ensemble, baseline, mean_pred):
    if mean_pred is not None:
        torch.save(mean_pred, os.path.join(output_path, f'{time_step}-regression-prediction'))
    torch.save(target, os.path.join(output_path, f'{time_step}-target'))
    torch.save(prediction_ensemble, os.path.join(output_path, f'{time_step}-predictions'))
    torch.save(baseline, os.path.join(output_path, f'{time_step}-baseline'))

# Takes templates from EvalML
def save_results_as_grib(output_path, time_step, target, prediction_ensemble, baseline, mean_pred,
    dataset, grib_template_path):

    # Somewhat kludgey way of getting the grid.
    if target.shape[1] == 352:
        grid='co2'
    else:
        grid='co1e'

    input_fields = dataset.input_channels()
    output_fields = dataset.output_channels()
    static_fields = dataset.static_channels()

    # Target
    output_file = os.path.join(output_path, f'{time_step}-target.grib')
    save_image_as_grib(output_file, time_step, grib_template_path, output_fields + static_fields, target, grid=grid)

    # Prediction - 1 file per ensemble?
    if len(prediction_ensemble.shape) == 4:
        for i in range(prediction_ensemble.shape[0]):
            output_file = os.path.join(output_path, f'{time_step}-pred{i:02}.grib')
            save_image_as_grib(output_file, time_step, grib_template_path, output_fields + static_fields, prediction_ensemble[i,:], grid=grid)
    else:
        # no ensemble dimension
        output_file = os.path.join(output_path, f'{time_step}-pred.grib')
        save_image_as_grib(output_file, time_step, grib_template_path, output_fields + static_fields, prediction_ensemble, grid=grid)

    # Baseline
    output_file = os.path.join(output_path, f'{time_step}-baseline.grib')
    save_image_as_grib(output_file, time_step, grib_template_path, input_fields, baseline, grid=grid)

    return


def save_image_as_grib(output_filename, time_step, grib_template_path, channels, image, grid):
    if grid == "co2":
        padding_margin = 19
    elif grid == "co1e":
        padding_margin = 41
    else:
        raise ValueError("only co1e and co2 grid supported")

    ds_r = ekd.FieldList()
    for i in range(len(channels)):
        channel = channels[i]

        # raise ValueError if not found
        ds = get_grib_template(grib_template_path, channel, time_step, grid)
        if ds:
            md_new = ds.metadata()
            values = pad_image(image[i,::], padding_margin, np.nan)
            ds_new = ekd.FieldList.from_array(values, md_new)
            ds_r += ds_new
    #output_file = os.path.join(output_path, f'{time_step}.grib')
    # Metadata is shown (correctly) as different channels here.
    logging.debug(f'ds_r is {ds_r.ls()}')
    # Metadata is not propagated, for some reason-- all channels have same metadata as ds[0] (2t).
    # However, values are preserved properly.
    with ekd.create_target("file",output_filename) as t:
        for f in ds_r:
            t.write(f)

# grid: co2 (COSMO-2), or co1e (COSMO-1E)
def get_grib_template(grib_template_path, channel, datetime, grid="co2"):
    [date,time]=datetime.split('-')
    # Get index of the channels types
    levtype_index_sfc = ekd.from_source("file", os.path.join(grib_template_path, "ifs-levtype=sfc.grib"))
    levtype_index_pl = ekd.from_source("file", os.path.join(grib_template_path, "ifs-levtype=pl.grib"))
    if channel.name == 'tp':
        ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-shortName=TOT_PREC.grib'))
        return ds[0].clone(shortName=channel.name, dataDate=date, dataTime=time)
    elif channel.name in levtype_index_sfc.metadata("shortName") and (channel.level==None or channel.level=='' or int(channel.level) < 50):
        idx = levtype_index_sfc.metadata("shortName").index(channel.name)
        levtype = levtype_index_sfc[idx].metadata("typeOfLevel")
        levelval = levtype_index_sfc[idx].metadata("level")
        try:
            ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-typeOfLevel={levtype}.grib'))
        except FileNotFoundError as e:
            logging.warning(f'Channel {channel.name} not found in GRIB templates: {e}')
            logging.warning(f'Skipping channel {channel.name}')
            return None
        return ds[0].clone(shortName=channel.name, level=levelval, dataDate=date, dataTime=time)
    elif channel.name in levtype_index_pl.metadata("shortName") and channel.level:
        levtype='isobaricInhPa'
        ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-typeOfLevel={levtype}.grib'))
        return ds[0].clone(shortName=channel.name, level=channel.level, dataDate=date, dataTime=time) 
    else:
        logging.warning(f'channel {channel.name} not found in grib index; skipping')
        return None


def pad_image(image, padding_margin, fill_value):
    new_image = np.ones(((image.shape[0] + padding_margin * 2), (image.shape[1] + padding_margin * 2))) * fill_value
    new_image[padding_margin:-padding_margin, padding_margin:-padding_margin] = image
    return new_image

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

    # Prediction - 1 file per ensemble?
    if len(prediction_ensemble.shape) == 4:
        for i in range(prediction_ensemble.shape[0]):
            output_file = os.path.join(output_path, f'{time_step}-pred{i:02}.grib')
            save_image_as_grib(output_file, time_step, grib_template_path, output_fields, prediction_ensemble[i,:], grid=grid)
    else:
        # no ensemble dimension
        output_file = os.path.join(output_path, f'{time_step}-pred.grib')
        save_image_as_grib(output_file, time_step, grib_template_path, output_fields, prediction_ensemble, grid=grid)

    # Baseline
    output_file = os.path.join(output_path, f'{time_step}-baseline.grib')
    save_image_as_grib(output_file, time_step, grib_template_path, input_fields, baseline, grid=grid)

    # EvalML-compatible output: one file per init time, step 000
    # Use mean prediction (regression) if available, otherwise first ensemble member
    evalml_data = mean_pred if mean_pred is not None else (
        prediction_ensemble[0] if len(prediction_ensemble.shape) == 4 else prediction_ensemble
    )
    init_time = time_step.replace('-', '')
    evalml_file = os.path.join(output_path, f'{init_time}_000.grib')
    save_image_as_grib(evalml_file, time_step, grib_template_path, output_fields, evalml_data, grid=grid)

    return


def save_image_as_grib(output_filename, time_step, grib_template_path, channels, image, grid):
    if grid == "co2":
        padding_margin = 19
    elif grid == "co1e":
        padding_margin = 41
    else:
        raise ValueError("only co1e and co2 grid supported")

    with open(output_filename, 'wb') as f_out:
        for i, channel in enumerate(channels):
            result = get_grib_template(grib_template_path, channel, time_step, grid)
            if result is None:
                continue
            template_field, grib_keys = result
            values = pad_image(image[i, ::], padding_margin, np.nan)
            grib_id = eccodes.codes_new_from_message(template_field.message())
            try:
                for key, val in grib_keys.items():
                    eccodes.codes_set(grib_id, key, val)
                flat = values.flatten().astype(float)
                missing = 9999.0
                eccodes.codes_set(grib_id, 'bitmapPresent', 1)
                eccodes.codes_set(grib_id, 'missingValue', missing)
                flat[np.isnan(flat)] = missing
                eccodes.codes_set_values(grib_id, flat)
                eccodes.codes_write(grib_id, f_out)
            finally:
                eccodes.codes_release(grib_id)

# grid: co2 (COSMO-2), or co1e (COSMO-1E)
# Returns (template_field, grib_keys_dict) or None if channel has no template.
# grib_keys are applied via eccodes after codes_new_from_message to avoid
# earthkit clone() silently dropping key overrides.
def get_grib_template(grib_template_path, channel, datetime, grid="co2"):
    [date, time] = datetime.split('-')
    date, time = int(date), int(time)
    levtype_index_sfc = ekd.from_source("file", os.path.join(grib_template_path, "ifs-levtype=sfc.grib"))
    levtype_index_pl = ekd.from_source("file", os.path.join(grib_template_path, "ifs-levtype=pl.grib"))
    if channel.name == 'tp':
        ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-shortName=TOT_PREC.grib'))
        return ds[0], {'dataDate': date, 'dataTime': time}
    elif channel.name in levtype_index_sfc.metadata("shortName") and (channel.level==None or channel.level=='' or int(channel.level) < 50):
        idx = levtype_index_sfc.metadata("shortName").index(channel.name)
        levtype = levtype_index_sfc[idx].metadata("typeOfLevel")
        levelval = levtype_index_sfc[idx].metadata("level")
        param_id = levtype_index_sfc[idx].metadata("paramId")
        try:
            ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-typeOfLevel={levtype}.grib'))
        except FileNotFoundError as e:
            logging.warning(f'Channel {channel.name} not found in GRIB templates: {e}')
            logging.warning(f'Skipping channel {channel.name}')
            return None
        return ds[0], {'paramId': param_id, 'level': levelval, 'dataDate': date, 'dataTime': time}
    elif channel.name in levtype_index_pl.metadata("shortName") and channel.level:
        levtype='isobaricInhPa'
        idx = levtype_index_pl.metadata("shortName").index(channel.name)
        param_id = levtype_index_pl[idx].metadata("paramId")
        ds = ekd.from_source("file", os.path.join(grib_template_path, f'{grid}-typeOfLevel={levtype}.grib'))
        return ds[0], {'paramId': param_id, 'level': int(channel.level), 'dataDate': date, 'dataTime': time}
    else:
        logging.warning(f'channel {channel.name} not found in grib index; skipping')
        return None


def pad_image(image, padding_margin, fill_value):
    new_image = np.ones(((image.shape[0] + padding_margin * 2), (image.shape[1] + padding_margin * 2))) * fill_value
    new_image[padding_margin:-padding_margin, padding_margin:-padding_margin] = image
    return new_image


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
    