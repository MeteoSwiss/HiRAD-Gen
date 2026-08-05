import hydra
import os
import json
import time
from collections import defaultdict
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
from hirad.utils.inference_utils import save_results
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape

def _sync_t() -> float:
    """Return wall-clock time after synchronizing all pending CUDA ops."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


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

    # Set precision for inference
    input_dtype = torch.float16 if cfg.generation.perf.get("force_fp16", False) else torch.float32

    # Parse the inference input times
    if cfg.generation.get("times_range", None) and cfg.generation.get("times", None):
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.get("times_range", None):
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") #TODO check what time formats we are using and adapt
    elif cfg.generation.get("times", None):
        times = cfg.generation.times
    else:
        raise ValueError("Either times_range or times must be provided")

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler_inference(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    dataset.stats_to_torch(device=dist.device, dtype=input_dtype)
    dataset.interpolator.to(device=dist.device)
    is_real_target = dataset_cfg.get("type").split("_")[-1] == "real"
    if is_real_target:
        dataset.regrid_indices_real = dataset.regrid_indices_real.to(dist.device)
        dataset.regrid_weights_real = dataset.regrid_weights_real.to(dist.device, dtype=input_dtype)      
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())


    #TODO: Isolate loading into the method of generator
    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    use_apex_gn = True
    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_path = cfg.generation.io.res_ckpt_path
        logger0.info(f'Loading correction network from "{res_ckpt_path}"...')

        diffusion_model_args_path = os.path.join(res_ckpt_path, 'model_args.json')
        if not os.path.isfile(diffusion_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{diffusion_model_args_path}'.")
        with open(diffusion_model_args_path, 'r') as f:
            diffusion_model_args = json.load(f)
        # Disable AMP for inference (even if model is trained with AMP)
        if "amp_mode" in diffusion_model_args:
            diffusion_model_args["amp_mode"] = False
        use_apex_gn = diffusion_model_args.get("use_apex_gn", False)  # TODO: restore once apex available
        #use_apex_gn = False
        #diffusion_model_args["use_apex_gn"] = use_apex_gn

        net_res = EDMPrecondSuperResolution(**diffusion_model_args)

        _ = load_checkpoint(
            path=res_ckpt_path,
            model=net_res,
            device=dist.device
        )
        
        net_res = net_res.eval().to(device)
        if use_apex_gn:
            net_res = net_res.to(memory_format=torch.channels_last)
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
        # Disable AMP for inference (even if model is trained with AMP)
        if "amp_mode" in regression_model_args:
            regression_model_args["amp_mode"] = False
        use_apex_gn_reg = regression_model_args.get("use_apex_gn", False)  # TODO: restore once apex available
        #use_apex_gn_reg = False
        #regression_model_args["use_apex_gn"] = use_apex_gn_reg

        net_reg = UNet(**regression_model_args)

        _ = load_checkpoint(
            path=reg_ckpt_path,
            model=net_reg,
            device=dist.device
        )
        
        net_reg = net_reg.eval().to(device)
        if use_apex_gn_reg:
            net_reg = net_reg.to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)



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
    sampler_params = dict(OmegaConf.to_container(cfg.sampler.params, resolve=True)) if "params" in cfg.sampler else {}
    sampler_params["use_apex_gn"] = use_apex_gn
    generator.initialize_sampler(cfg.sampler.type, **sampler_params)
    
    # generate images
    _io_cfg = getattr(cfg.generation, "io", None)
    output_path = getattr(_io_cfg, "output_path", "./outputs")
    output_format = getattr(_io_cfg, "output_format", "torch")
    if output_format not in ['torch', 'grib', 'both']:
        raise ValueError(f'Invalid output format {output_format}, must be \'torch\', \'grib\' or \'both\'')
    grib_template_path = getattr(_io_cfg, "grib_template_path", "")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)
    enable_timing = cfg.generation.perf.get("enable_timing", True)
    _t = _sync_t if enable_timing else (lambda: 0.0)

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available() and cfg.generation.perf.get("profile", False)
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available() and cfg.generation.perf.get("profile", False)
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:
            with torch.inference_mode():

                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True,
                    num_workers=4, persistent_workers=True,
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

                # Per-section timing accumulators (wall-clock, GPU-synchronized)
                step_timings = defaultdict(float)
                timed_step_count = 0

                #TODO: Isolate static channel loading into the method of generator or reuse training manager static channel loading
                static_channels = dataset.get_static_data()
                if static_channels is not None:
                    static_channels = static_channels[None, ::].flip(-2)
                    if use_apex_gn:
                        static_channels = static_channels.to(
                            dist.device,
                            dtype=input_dtype,
                            non_blocking=True,
                        ).to(memory_format=torch.channels_last)
                    else:
                        static_channels = (
                            static_channels.to(dist.device)
                            .to(input_dtype)
                            .contiguous()
                        )
                lead_time_label = None

                times = dataset.time()
                # t_iter_end is updated at the end of each loop body; the gap between
                # t_iter_end[i] and the start of body[i+1] equals DataLoader fetch time.
                t_iter_end = _t()
                for index, (image_tar, image_lr, *date_str) in enumerate(
                    iter(data_loader)
                ):
                    t_iter_start = _t()
                    t_data_load = t_iter_start - t_iter_end

                    t_preproc_start = _t()
                    
                    time_index += 1
                    if dist.rank == 0:
                        logger0.info(f"starting index: {time_index} time: {times[sampler[time_index]]}")

                    if time_index == warmup_steps:
                        start.record()

                    savedir = os.path.join(output_path,f"{times[sampler[time_index]]}")
                    os.makedirs(savedir,exist_ok=True)

                    #TODO: Move all the data processing inside the generator and just pass raw data to it. This includes regridding, normalization, date embedding creation, etc.
                    # Same as with static channel loading, we can reuse some of the code from training manager for this. This will also make it easier to maintain and update the data processing steps in one place.
                    if is_real_target:
                        image_tar = regrid_icon_to_rotlatlon(
                            image_tar.to(dist.device, dtype=input_dtype),
                            dataset.regrid_indices_real,
                            dataset.regrid_weights_real,
                        )
                        if dataset.trim_edge > 0:
                            image_tar = image_tar[:, :, dataset.trim_edge:-dataset.trim_edge, dataset.trim_edge:-dataset.trim_edge]
                        #image_tar = image_tar.flip(-2)  # May be needed
                    else:
                        image_tar = image_tar.reshape(*image_tar.shape[:-1], *dataset.image_shape())
                    if lead_time_label:
                        lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                    else:
                        lead_time_label = None
                    image_lr = dataset.interpolator(image_lr.to(dist.device, dtype=input_dtype)).reshape(*image_lr.shape[:-1], *dataset.image_shape()).flip(-2)
                    image_lr = dataset.normalize_input(image_lr)
                    if use_apex_gn:
                        image_lr = image_lr.to(memory_format=torch.channels_last)
                    date_embedding = None
                    if dataset._n_month_hour_channels:
                        date_embedding = dataset.make_time_grids(*date_str, dist.device, dtype=input_dtype)

                    random_seed = cfg.generation.get("random_seed", None)+index if cfg.generation.get("randomize", False) and cfg.generation.get("random_seed", None) is not None else None
                    t_preproc_end = _t()

                    t_gen_start = _t()
                    image_out, image_reg = generator.generate(
                                                image_lr,
                                                static_channels=static_channels,
                                                date_embedding=date_embedding,
                                                lead_time_label=lead_time_label,
                                                randomize=cfg.generation.get("randomize", False),
                                                random_seed=random_seed,
                                                skip_timing=(not enable_timing or time_index < warmup_steps),
                                            )
                    t_gen_end = _t()

                    t_postproc_start = _t()
                    if dist.rank == 0:
                        batch_size = image_out.shape[0]
                        # write out data in a seperate thread so we don't hold up inferencing
                        image_tar = image_tar[0].squeeze().cpu().numpy()
                        prediction_ensemble = dataset.denormalize_output(image_out).squeeze().flip(-2).cpu().numpy()
                        baseline = dataset.denormalize_input(image_lr)[0].squeeze().flip(-2).cpu().numpy()
                        if image_reg is not None:
                            mean_pred = dataset.denormalize_output(image_reg)[0].squeeze().flip(-2).cpu().numpy()
                    t_postproc_end = _t()

                    t_write_start = _t()
                    if dist.rank == 0:
                        writer_threads.append(
                            writer_executor.submit(
                                save_results,
                                savedir,
                                times[sampler[time_index]],
                                dataset,
                                prediction_ensemble,
                                image_tar,
                                baseline,
                                mean_pred if image_reg is not None else None,
                                output_format=output_format,
                                grib_template_path=grib_template_path,
                            )
                        )
                    t_write_end = _t()

                    if enable_timing and time_index >= warmup_steps:
                        timed_step_count += 1
                        step_timings["data_loading"] += t_data_load
                        step_timings["preprocessing"] += t_preproc_end - t_preproc_start
                        step_timings["generation"] += t_gen_end - t_gen_start
                        step_timings["postprocessing"] += t_postproc_end - t_postproc_start
                        step_timings["io_submit"] += t_write_end - t_write_start

                    t_iter_end = _t()

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

                # Log per-section timing breakdown
                if dist.rank == 0 and timed_step_count > 0:
                    logger0.info("--- Inference timing breakdown (avg over timed steps, wall-clock GPU-synced) ---")
                    for key in ["data_loading", "preprocessing", "generation", "postprocessing", "io_submit"]:
                        avg = step_timings[key] / timed_step_count
                        logger0.info(f"  {key:20s}: {avg:.3f} s/step")
                    # Log generator's internal breakdown (regression / diffusion / gather)
                    gen_timings = generator.get_timings()
                    if gen_timings:
                        logger0.info("--- Generator internal timing breakdown (avg over timed steps, warmup excluded) ---")
                        for key, (total, count) in gen_timings.items():
                            avg = total / count if count > 0 else 0.0
                            if key.endswith("_calls"):
                                logger0.info(f"  {key:20s}: {avg:.1f} calls/step  (n={count})")
                            else:
                                logger0.info(f"  {key:20s}: {avg:.3f} s/step  (n={count})")
                        # Derived: average time per individual net() forward call
                        if "diff_net_forward" in gen_timings and "diff_net_forward_calls" in gen_timings:
                            fwd_total, fwd_count = gen_timings["diff_net_forward"]
                            calls_total, calls_count = gen_timings["diff_net_forward_calls"]
                            if calls_total > 0:
                                per_call_ms = (fwd_total / calls_total) * 1000
                                logger0.info(f"  {'diff_net_fwd/call':20s}: {per_call_ms:.1f} ms/call")

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