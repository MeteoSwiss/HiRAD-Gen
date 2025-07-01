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

from hirad.models import EDMPrecondSuperResolution, UNet
from hirad.inference import Generator
from hirad.utils.inference_utils import save_images
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random dowscaled atmospheric states using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    # torch.backends.cudnn.enabled = False
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)

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
    
    generator = Generator(
        net_reg=net_reg,
        net_res=net_res,
        batch_size=cfg.generation.seed_batch_size,
        ensemble_size=cfg.generation.num_ensembles,
        hr_mean_conditioning=cfg.generation.hr_mean_conditioning,
        n_out_channels=img_out_channels,
        inference_mode=cfg.generation.inference_mode,
        dist=dist,
        )

    # Parse the patch shape
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        generator.initialize_patching(img_shape=img_shape, 
                                      patch_shape=patch_shape,
                                      boundary_pix=cfg.generation.boundary_pix,
                                      overlap_pix=cfg.generation.overlap_pix,
                                      )
    sampler_params = cfg.sampler.params if "params" in cfg.sampler else {}
    generator.initialize_sampler(cfg.sampler.type, **sampler_params)
    
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
                # image_out, image_reg = generate_fn(image_lr,lead_time_label)
                image_out, image_reg = generator.generate(image_lr,lead_time_label)
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


if __name__ == "__main__":
    main()