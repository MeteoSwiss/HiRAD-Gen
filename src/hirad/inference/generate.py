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

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from einops import rearrange
from torch.distributed import gather


from hydra.utils import to_absolute_path
from hirad.models import EDMPrecondSR, UNet
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
        logger0.info(f'Loading correction network from "{res_ckpt_path}"...')

        diffusion_model_args_path = os.path.join(res_ckpt_path, 'model_args.json')
        if not os.path.isfile(diffusion_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{diffusion_model_args_path}'.")
        with open(diffusion_model_args_path, 'r') as f:
            diffusion_model_args = json.load(f)

        net_res = EDMPrecondSR(**diffusion_model_args)

        _ = load_checkpoint(
            path=res_ckpt_path,
            model=net_res,
            device=dist.device
        )
        
        #TODO fix to use channels_last which is optimal for H100
        net_res = net_res.eval().to(device)#.to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
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
        
        net_reg = net_reg.eval().to(device)#.to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
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
        sampler_fn = partial(
            stochastic_sampler,
            img_shape=img_shape,
            patch_shape_x=patch_shape[0],
            patch_shape_y=patch_shape[1],
            boundary_pix=cfg.sampler.boundary_pix,
            overlap_pix=cfg.sampler.overlap_pix,
        )
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")
    

        # Main generation definition
    def generate_fn(image_lr, labels, lead_time_label):
        img_shape_y, img_shape_x = img_shape
        with nvtx.annotate("generate_fn", color="green"):
            if cfg.generation.sample_res == "full":
                image_lr_patch = image_lr
            else:
                torch.cuda.nvtx.range_push("rearrange")
                image_lr_patch = rearrange(
                    image_lr,
                    "b c (h1 h) (w1 w) -> (b h1 w1) c h w",
                    h1=img_shape_y // patch_shape[0],
                    w1=img_shape_x // patch_shape[1],
                )
                torch.cuda.nvtx.range_pop()
            image_lr_patch = image_lr_patch #.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=image_lr_patch,
                        labels=labels,
                        latents_shape=(
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),
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
                        seed_batch_size=cfg.generation.seed_batch_size,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=image_lr_patch.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ), #.to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        hr_mean=mean_hr,
                        lead_time_label=lead_time_label,
                    )
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg[0:1,::] + image_res

            if cfg.generation.sample_res != "full":
                image_out = rearrange(
                    image_out,
                    "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
                    h1=img_shape_y // patch_shape[0],
                    w1=img_shape_x // patch_shape[1],
                )
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
                    return torch.cat(gathered_tensors)
                else:
                    return None
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

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer_executor = ThreadPoolExecutor(
                    max_workers=cfg.generation.perf.num_writer_workers
                )
                writer_threads = []

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            times = dataset.time()
            for image_tar, image_lr, labels, *lead_time_label in iter(data_loader):
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
                    #.to(memory_format=torch.channels_last)
                )
                image_tar = image_tar.to(device=device).to(torch.float32)
                labels = labels.to(device).to(torch.float32).contiguous()
                image_out, image_reg = generate_fn(image_lr,labels,lead_time_label)
                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    # write out data in a seperate thread so we don't hold up inferencing
                    writer_threads.append(
                        writer_executor.submit(
                            save_images,
                            output_path,
                            times[sampler[time_index]],
                            dataset,
                            image_out.cpu(),
                            image_tar.cpu(),
                            image_lr.cpu(),
                            image_reg.cpu() if image_reg is not None else None,
                        )
                    )
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000.0  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0:
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
    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()
    image_pred = image_pred.numpy()
    image_pred_final = np.flip(dataset.denormalize_output(image_pred[-1,::].squeeze()),1).reshape(len(output_channels),-1)
    if image_pred.shape[0]>1:
        image_pred_mean = np.flip(dataset.denormalize_output(image_pred.mean(axis=0)),1).reshape(len(output_channels),-1)
        image_pred_first_step = np.flip(dataset.denormalize_output(image_pred[0,::].squeeze()),1).reshape(len(output_channels),-1)
        image_pred_mid_step = np.flip(dataset.denormalize_output(image_pred[image_pred.shape[0]//2,::].squeeze()),1).reshape(len(output_channels),-1)
    image_hr = np.flip(dataset.denormalize_output(image_hr[0,::].squeeze().numpy()),1).reshape(len(output_channels),-1)
    image_lr = np.flip(dataset.denormalize_input(image_lr[0,::].squeeze().numpy()),1).reshape(len(input_channels),-1)
    if mean_pred is not None:
        mean_pred = np.flip(dataset.denormalize_output(mean_pred[0,::].squeeze().numpy()),1).reshape(len(output_channels),-1)
    os.makedirs(output_path, exist_ok=True)
    for idx, channel in enumerate(output_channels):
        input_channel_idx = input_channels.index(channel)
        _plot_projection(longitudes,latitudes,image_lr[input_channel_idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-lr.jpg'))
        _plot_projection(longitudes,latitudes,image_hr[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-hr.jpg'))
        _plot_projection(longitudes,latitudes,image_pred_final[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-hr-pred.jpg'))
        if image_pred.shape[0]>1:
            _plot_projection(longitudes,latitudes,image_pred_mean[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-hr-pred-mean.jpg'))
            _plot_projection(longitudes,latitudes,image_pred_first_step[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-hr-pred-0.jpg'))
            _plot_projection(longitudes,latitudes,image_pred_mid_step[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-hr-pred-mid.jpg'))
        if mean_pred is not None:
            _plot_projection(longitudes,latitudes,mean_pred[idx,:],os.path.join(output_path,f'{time_step}-{channel.name}-mean-pred.jpg'))

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
    