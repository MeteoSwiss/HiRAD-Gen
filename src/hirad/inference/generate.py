import hydra
import os
import json
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import nvtx
import numpy as np
import contextlib

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from torch.distributed import gather

from hirad.models import EDMPrecondSuperResolution, UNet
from hirad.utils.patching import GridPatching2D
from hirad.inference import stochastic_sampler
from hirad.inference import deterministic_sampler
from hirad.utils.inference_utils import (
    regression_step,
    diffusion_step,
)
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape

from hirad.eval import compute_mae, average_power_spectrum, plot_error_projection, plot_power_spectra

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random dowscaled atmospheric states using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    torch.backends.cudnn.enabled = False
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)

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
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") #TODO check what time formats we are using and adapt
    else:
        times = cfg.generation.times

    # Create dataset object
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

    # Parse the patch shape
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
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
        logger0.info(f'Loading correction network from "{res_ckpt_path}"...')

        diffusion_model_args_path = os.path.join(res_ckpt_path, 'model_args.json')
        if not os.path.isfile(diffusion_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{diffusion_model_args_path}'.")
        with open(diffusion_model_args_path, 'r') as f:
            diffusion_model_args = json.load(f)

        net_res = EDMPrecondSuperResolution(**diffusion_model_args)

        _ = load_checkpoint(
            path=res_ckpt_path,
            model=net_res,
            device=dist.device
        )
        
        #TODO fix to use channels_last which is optimal for H100
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True

        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_path = cfg.generation.io.reg_ckpt_path
        logger0.info(f'Loading regression network from "{reg_ckpt_path}"...')


        regression_model_args_path = os.path.join(reg_ckpt_path, 'model_args.json')
        if not os.path.isfile(regression_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{regression_model_args_path}'.")
        with open(regression_model_args_path, 'r') as f:
            regression_model_args = json.load(f)

        net_reg = UNet(**regression_model_args)

        _ = load_checkpoint(
            path=reg_ckpt_path,
            model=net_reg,
            device=dist.device
        )
        
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True

        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

        # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        # Only compile residual network
        # Overhead of compiling regression network outweights any benefits
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")

    # Partially instantiate the sampler based on the configs
    if cfg.sampler.type == "deterministic":
        if cfg.generation.hr_mean_conditioning:
            raise NotImplementedError(
                "High-res mean conditioning is not yet implemented for the deterministic sampler"
            )
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            # num_ensembles=cfg.generation.num_ensembles,
            solver=cfg.sampler.solver,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(stochastic_sampler, patching=patching)
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")
    

        # Main generation definition
    def generate_fn(image_lr, lead_time_label):
        with nvtx.annotate("generate_fn", color="green"):
            # (1, C, H, W)
            image_lr = image_lr.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=image_lr,
                        latents_shape=(
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ), # (batch_size, C, H, W)
                        lead_time_label=lead_time_label,
                    )
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=image_lr.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ), #.to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                    )
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg[0:1,::] + image_res

            # Gather tensors on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if dist.rank == 0 else None,
                    dst=0,
                )

                if dist.rank == 0:
                    if cfg.generation.inference_mode != "regression":
                        return torch.cat(gathered_tensors), image_reg[0:1,::]
                    return torch.cat(gathered_tensors), None
                else:
                    return None, None
            else:
                #TODO do this for multi-gpu setting above too
                if cfg.generation.inference_mode != "regression":
                    return image_out, image_reg
                return image_out, None
    
    # generate images
    output_path = getattr(cfg.generation.io, "output_path", "./outputs")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)
    # Generates model predictions from the input data using the specified
    # `generate_fn`, and save the predictions to the provided NetCDF file. It iterates
    # through the dataset using a data loader, computes predictions, and saves them along
    # with associated metadata.

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer_executor = ThreadPoolExecutor(
                    max_workers=cfg.generation.perf.num_writer_workers
                )
                writer_threads = []

            # Create timer objects only if CUDA is available
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            else:
                # Dummy no-op functions for CPU case
                class DummyEvent:
                    def record(self):
                        pass

                    def synchronize(self):
                        pass

                    def elapsed_time(self, _):
                        return 0

                start = end = DummyEvent()

            times = dataset.time()
            for index, (image_tar, image_lr, *lead_time_label) in enumerate(
                iter(data_loader)
            ):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                # continue
                if lead_time_label:
                    lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                else:
                    lead_time_label = None
                image_lr = (
                    image_lr.to(device=device)
                    .to(torch.float32)
                    .to(memory_format=torch.channels_last)
                )
                image_tar = image_tar.to(device=device).to(torch.float32)
                image_out, image_reg = generate_fn(image_lr,lead_time_label)
                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    # write out data in a seperate thread so we don't hold up inferencing
                    writer_threads.append(
                        writer_executor.submit(
                            save_images,
                            output_path,
                            times[sampler[time_index]],
                            dataset,
                            image_out.cpu().numpy(),
                            image_tar.cpu().numpy(),
                            image_lr.cpu().numpy(),
                            image_reg.cpu().numpy() if image_reg is not None else None,
                        )
                    )
            end.record()
            end.synchronize()
            elapsed_time = (
                start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0
            )  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(
                    f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


def save_images(output_path, time_step, dataset, image_pred, image_hr, image_lr, mean_pred):
    
    os.makedirs(output_path, exist_ok=True)

    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()

    target = np.flip(dataset.denormalize_output(image_hr[0,::].squeeze()),1) #.reshape(len(output_channels),-1)
    prediction = np.flip(dataset.denormalize_output(image_pred[-1,::].squeeze()),1) #.reshape(len(output_channels),-1)
    baseline = np.flip(dataset.denormalize_input(image_lr[0,::].squeeze()),1)# .reshape(len(input_channels),-1) 
    if mean_pred is not None:
        mean_pred = np.flip(dataset.denormalize_output(mean_pred[0,::].squeeze()),1) #.reshape(len(output_channels),-1)


    freqs = {}
    power = {}
    for idx, channel in enumerate(output_channels):
        input_channel_idx = input_channels.index(channel)

        if channel.name=="tp":
            target[idx,::] = prepare_precipitaiton(target[idx,:,:])
            prediction[idx,::] = prepare_precipitaiton(prediction[idx,:,:])
            baseline[input_channel_idx,:,:] = prepare_precipitaiton(baseline[input_channel_idx])
            if mean_pred is not None:
                mean_pred[idx,::] = prepare_precipitaiton(mean_pred[idx,::])

        _plot_projection(longitudes, latitudes, target[idx,:,:], os.path.join(output_path, f'{time_step}-{channel.name}-target.jpg'))
        _plot_projection(longitudes, latitudes, prediction[idx,:,:], os.path.join(output_path, f'{time_step}-{channel.name}-prediction.jpg'))
        _plot_projection(longitudes, latitudes, baseline[input_channel_idx,:,:], os.path.join(output_path, f'{time_step}-{channel.name}-input.jpg'))
        if mean_pred is not None:
            _plot_projection(longitudes, latitudes, mean_pred[idx,:,:], os.path.join(output_path, f'{time_step}-{channel.name}-mean_prediction.jpg'))

        _, baseline_errors = compute_mae(baseline[input_channel_idx,:,:], target[idx,:,:])
        _, prediction_errors = compute_mae(prediction[idx,:,:], target[idx,:,:])
        if mean_pred is not None:
            _, mean_prediction_errors = compute_mae(mean_pred[idx,:,:], target[idx,:,:])


        plot_error_projection(baseline_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path, f'{time_step}-{channel.name}-baseline-error.jpg'))
        plot_error_projection(prediction_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path, f'{time_step}-{channel.name}-prediction-error.jpg'))
        if mean_pred is not None:
            plot_error_projection(mean_prediction_errors.reshape(-1), latitudes, longitudes, os.path.join(output_path, f'{time_step}-{channel.name}-mean-prediction-error.jpg'))

        b_freq, b_power = average_power_spectrum(baseline[input_channel_idx,:,:].squeeze(), 2.0)
        freqs['baseline'] = b_freq
        power['baseline'] = b_power
        #plotting.plot_power_spectrum(b_freq, b_power, target_channels[t_c], os.path.join('plots/spectra/baseline2dt',  target_channels[t_c] + '-all_dates'))
        t_freq, t_power = average_power_spectrum(target[idx,:,:].squeeze(), 2.0)
        freqs['target'] = t_freq
        power['target'] = t_power
        p_freq, p_power = average_power_spectrum(prediction[idx,:,:].squeeze(), 2.0)
        freqs['prediction'] = p_freq
        power['prediction'] = p_power
        if mean_pred is not None:
            mp_freq, mp_power = average_power_spectrum(mean_pred[idx,:,:].squeeze(), 2.0)
            freqs['mean_prediction'] = mp_freq
            power['mean_prediction'] = mp_power
        plot_power_spectra(freqs, power, channel.name, os.path.join(output_path, f'{time_step}-{channel.name}-spectra.jpg'))


def prepare_precipitaiton(precip_array):
    precip_array = np.clip(precip_array, 0, None)
    epsilon = 1e-2
    precip_array = precip_array + epsilon
    precip_array = np.log(precip_array)
    # log_min, log_max = precip_array.min(), precip_array.max()
    # precip_array = (precip_array-log_min)/(log_max-log_min)
    return precip_array


def _plot_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str, cmap=None, vmin = None, vmax = None):

    """Plot observed or interpolated data in a scatter plot."""
    # TODO: Refactor this somehow, it's not really generalizing well across variables.
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig(filename)
    plt.close('all')

if __name__ == "__main__":
    main()