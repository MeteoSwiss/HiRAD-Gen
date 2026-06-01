import os
import time

from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from contextlib import nullcontext
import nvtx
import numpy as np
import torch
from hydra.utils import to_absolute_path
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import mlflow
# from torchinfo import summary

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from hirad.utils.train_helpers import set_seed, configure_cuda_for_consistent_precision, \
                                        set_patch_shape, compute_num_accumulation_rounds, calculate_patch_per_iter, \
                                        is_time_for_periodic_task, handle_and_clip_gradients, \
                                        init_mlflow, update_learning_rate, log_training_progress, \
                                        cuda_profiler, cuda_profiler_start, cuda_profiler_stop, profiler_emit_nvtx
from hirad.utils.checkpoint import load_checkpoint, save_checkpoint
from hirad.utils.patching import RandomPatching2D
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.inference_utils import save_results_as_torch
from hirad.utils.env_info import get_env_info, flatten_dict
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon
from hirad.models import UNet, EDMPrecondSuperResolution
from hirad.losses import ResidualLoss, RegressionLoss
from hirad.datasets import init_train_valid_datasets_from_config, get_dataset_and_sampler_inference
from hirad.inference import Generator
from hirad.training.training_manager import TrainingManagerCorrDiff



torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264  # Set to a higher value
torch._dynamo.config.verbose = True  # Enable verbose logging
torch._dynamo.config.suppress_errors = False  # Forces the error to show all details
torch._logging.set_logs(recompiles=True, graph_breaks=True)


@hydra.main(version_base=None, config_path="../conf", config_name="training")
def main(cfg: DictConfig) -> None:

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    OmegaConf.resolve(cfg)

    # Initialize logging
    if cfg.logging.method == "mlflow":
        init_mlflow(cfg, dist)
        if dist.world_size > 1:
            torch.distributed.barrier()
    elif cfg.logging.method is not None:
        raise ValueError("The only available logging method is mlflow. To disable logging set the method to null.")

    logger = PythonLogger("main") # general logger
    logger0 = RankZeroLoggingWrapper(logger, dist) # rank 0 logger

    logger0.info(f"Config is: {cfg}")
    logger0.info(f"Saving the outputs in {os.getcwd()}")

    # create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(
        cfg.training.io.get("checkpoint_dir", "."), f"checkpoints_{cfg.model.name}"
    )
    if dist.rank==0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # performance optimization configuration
    use_torch_compile = getattr(cfg.training.perf, "torch_compile", False)
    use_apex_gn = getattr(cfg.training.perf, "use_apex_gn", False)
    profile_mode = getattr(cfg.training.perf, "profile_mode", False)
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    # set the data type for model inputs based on optimization configuration
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16
    
    # dataset configuration
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    train_test_split = getattr(cfg.dataset, "validation", False)
    n_month_hour_channels = 2*dataset_cfg.get("n_month_hour_channels", 0)
    
    # validate and set batch size configuration
    if cfg.training.hp.batch_size_per_gpu == "auto" and \
            cfg.training.hp.total_batch_size == "auto":
        raise ValueError("batch_size_per_gpu and total_batch_size can't be both set to 'auto'.")
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )
    elif cfg.training.hp.total_batch_size == "auto":
        cfg.training.hp.total_batch_size = (
            cfg.training.hp.batch_size_per_gpu * dist.world_size
        )

    # Get the current training step from the checkpoint if it exists, otherwise start from 0.
    cur_nimg = load_checkpoint(path=checkpoint_dir)

    # Fix the seed based on training progress for reproducibility.
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        train_test_split=train_test_split,
        sampler_start_idx=cur_nimg,
    )
    is_real_target = dataset_cfg.get("type").split("_")[-1].startswith("real")
    is_real2cosmo_target = dataset_cfg.get("type").split("_")[-1] == "real2cosmo"
    is_real2km_target = dataset_cfg.get("type").split("_")[-1] == "real2km"
    logger0.info(f"Training on dataset with size {len(dataset)}")
    logger0.info(f"Validating on dataset with size {len(validation_dataset) if validation_dataset else 0}")

    # Get the shape of the grid (without the channel dimension) for later use in model creation and patching
    img_shape = dataset.image_shape()

    logger0.info(f"Training on dataset with grid size {img_shape[0]}x{img_shape[1]}, {len(dataset.input_channels())} input channels and {len(dataset.output_channels())} output channels.")
    logger0.info(f"Input channels: {dataset.input_channels()}")
    logger0.info(f"Output channels: {dataset.output_channels()}")
    logger0.info(f"Static channels: {dataset.static_channels()}")

    # convert dataset stats to torch tensors on the correct device for later use in normalization and denormalization
    dataset.stats_to_torch(device=dist.device, dtype=input_dtype)
    # convert dataset stats to torch tensors on the correct device for later use in loss normalization and denormalization
    dataset.interpolator.to(device=dist.device)
    # convert regridding weights and indices to torch tensors on the correct device if real target dataset is used
    if is_real_target:
        dataset.regrid_indices_real = dataset.regrid_indices_real.to(dist.device)
        dataset.regrid_weights_real = dataset.regrid_weights_real.to(dist.device, dtype=input_dtype)

    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None

    # Parse the patch shape
    if (
        cfg.model.name == "patched_diffusion"
        or cfg.model.name == "lt_aware_patched_diffusion"
    ):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
    if (
        patch_shape_x
        and patch_shape_y
        and patch_shape_y >= img_shape[0]
        and patch_shape_x >= img_shape[1]
    ):
        logger0.warning(
            f"Patch shape {patch_shape_y}x{patch_shape_x} is larger than \
            the image shape {img_shape[0]}x{img_shape[1]}. Patching will not be used."
        )
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    
    # Initialize patcher if patch-based training is enabled
    if use_patching:
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")
    
    # Instantiate the training manager which handles model creation,
    # data loading and transformation, 
    # and validation
    training_manager = TrainingManagerCorrDiff(
                                            dist, 
                                            logger0, 
                                            dataset,
                                            input_dtype,
                                            img_shape,
                                            n_month_hour_channels, 
                                            fp16, 
                                            profile_mode, 
                                            enable_amp,
                                            amp_dtype,
                                            use_apex_gn,
                                            is_real_target,
                                            is_real2cosmo_target,
                                            is_real2km_target,
                                            songunet_checkpoint_level,
                                            use_patching,
                                            cfg.model.get("hr_mean_conditioning", False),
                                            cfg.logging.get("method", None)
                                            )

    # Create the model and move it to the appropriate device and memory format based on the optimization configuration
    model, model_args = training_manager.create_model(cfg.model.name, cfg.model.get("model_args", None))

    # # Print the model summary
    # if dist.rank == 0:
    #     summary(model, input_size=[(1, img_out_channels, *img_shape), (1, img_in_channels, *img_shape), (1,1)], device=dist.device)

    # raise NotImplementedError("Check if model_args are correct when using patching - img_in_channels should include global channels and lead time channels if applicable")

    
    model.train().requires_grad_(True).to(dist.device)

    if dist.rank==0 and not os.path.exists(os.path.join(checkpoint_dir, 'model_args.json')):
        with open(os.path.join(checkpoint_dir, f'model_args.json'), 'w') as f:
            json.dump(model_args, f)

    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    # Check if regression model is used with patching
    if (
        cfg.model.name in ["regression", "lt_aware_ce_regression"]
        and patching is not None
    ):
        raise ValueError(
            f"Regression model ({cfg.model.name}) cannot be used with patch-based training. "
        )

    # Load the regression checkpoint if applicable #TODO test when training correction
    regression_net = None
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_net = training_manager.load_regression_model(to_absolute_path(cfg.training.io.regression_checkpoint_path))


    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    # calculate patch per iter    
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    max_patch_per_gpu = getattr(cfg.training.hp, "max_patch_per_gpu", None)
    patch_nums_iter = calculate_patch_per_iter(patch_num, max_patch_per_gpu, batch_size_per_gpu)

    logger0.info(
        f"Patch number iterations are {patch_nums_iter}"
    )

    # Set patch gradient accumulation only for patched diffusion models
    if cfg.model.name in {
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    }:
        if len(patch_nums_iter) > 1:
            if not patching:
                logger0.info(
                    "Patching is not enabled: patch gradient accumulation automatically disabled."
                )
                use_patch_grad_acc = False
            else:
                use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    # Automatically disable patch gradient accumulation for non-patched models
    else:
        logger0.info(
            "Training a non-patched model: patch gradient accumulation automatically disabled."
        )
        use_patch_grad_acc = None


    # Instantiate the loss function
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        loss_fn = ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss()

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=cfg.training.hp.lr, 
        betas=[0.9, 0.999], 
        eps=1e-8,
        fused=True,
    )

    # Load optimizer checkpoint if it exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            device=dist.device,
        )
    except:
        cur_nimg = 0

    # Compile the model and regression net if applicable
    if use_torch_compile:
        # if dist.world_size==1:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        # if use_torch_compile:
        #     model = torch.compile(model)
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,  # dist.find_unused_parameters,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )


    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg

    # prepare static channels if there are any
    static_channels = training_manager.get_static_data()

    # turn off for lead time labels for now since we are not using them
    # TODO: implement lead time labels properly once we train on IFS?
    lead_time_label = None

    # enable profiler:
    with cuda_profiler():
        with profiler_emit_nvtx():
            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Starting Profiler at {cur_nimg}")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Stopping Profiler at {cur_nimg}")
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    # Compute & accumulate gradients
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0
                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(
                            f"accumulation round {n_i}", color="Magenta"
                        ):
                            with nvtx.annotate("loading data", color="green"):
                                tick_read_start_time = time.time()
                                img_clean, img_lr, date_embedding = training_manager.load_and_preprocess_batch(dataset_iterator)
                                tick_read_time = time.time() - tick_read_start_time
                            loss_fn_kwargs = {
                                "net": model,
                                "img_clean": img_clean,
                                "img_lr": img_lr,
                                "static_channels": static_channels,
                                "date_embedding": date_embedding,
                                "augment_pipe": None,
                                "use_apex_gn": use_apex_gn,
                            }
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs[
                                    "use_patch_grad_acc"
                                ] = use_patch_grad_acc

                            if lead_time_label:
                                lead_time_label = (
                                    lead_time_label[0].to(dist.device).contiguous()
                                )
                                loss_fn_kwargs.update(
                                    {"lead_time_label": lead_time_label}
                                )
                            else:
                                lead_time_label = None
                            if use_patch_grad_acc:
                                loss_fn.y_mean = None

                            for patch_num_per_iter in patch_nums_iter:
                                if patching is not None:
                                    patching.set_patch_num(patch_num_per_iter)
                                    loss_fn_kwargs.update({"patching": patching})
                                with nvtx.annotate(f"loss forward", color="green"):
                                    with torch.autocast(
                                        "cuda", dtype=amp_dtype, enabled=enable_amp
                                    ):
                                        loss = loss_fn(**loss_fn_kwargs)

                                loss = loss.sum() / batch_size_per_gpu / patch_num_per_iter
                                loss_accum += (
                                    loss
                                    / num_accumulation_rounds
                                    / len(patch_nums_iter)
                                )
                                with nvtx.annotate(f"loss backward", color="yellow"):
                                    loss.backward()

        
                    with nvtx.annotate(f"loss aggregate", color="green"):
                        loss_sum = torch.tensor([loss_accum], device=dist.device)
                        if dist.world_size > 1:
                            torch.distributed.barrier()
                            torch.distributed.all_reduce(
                                loss_sum, op=torch.distributed.ReduceOp.SUM
                            )
                        average_loss = (loss_sum / dist.world_size).cpu().item()

                        # update running mean of average loss since last periodic task
                        average_loss_running_mean += (
                            average_loss - average_loss_running_mean
                        ) / n_average_loss_running_mean
                        n_average_loss_running_mean += 1

                    # Update weights.
                    with nvtx.annotate("update weights", color="blue"):

                        current_lr = update_learning_rate(optimizer, 
                                                          cfg.training.hp.lr,
                                                          cfg.training.hp.lr_rampup,
                                                          cfg.training.hp.lr_decay,
                                                          cfg.training.hp.lr_decay_rate,
                                                          cur_nimg) 
                        handle_and_clip_gradients(
                            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
                        )
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration

                    # Logging training progress
                    if is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.print_progress_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    ):
                        # Print stats if we crossed the printing threshold with this batch
                        log_training_progress(logger0, cfg.logging.method, dist, cur_nimg, tick_start_nimg, tick_start_time,
                                              tick_read_time, start_time, average_loss, average_loss_running_mean, current_lr)
                        # reset running mean of average loss
                        average_loss_running_mean = 0
                        n_average_loss_running_mean = 1


                # Validation
                with nvtx.annotate("validation", color="red"):
                    if validation_dataset_iterator is not None and is_time_for_periodic_task(
                                                                        cur_nimg,
                                                                        cfg.training.io.validation_freq,
                                                                        done,
                                                                        cfg.training.hp.total_batch_size,
                                                                        dist.rank,
                                                                    ):
                        training_manager.run_validation(cur_nimg, validation_dataset_iterator, model, loss_fn, 
                                                        cfg.training.io.get("validation_steps",1), static_channels,
                                                        batch_size_per_gpu, patching, patch_nums_iter, use_patch_grad_acc)


                # Save checkpoints
                if dist.world_size > 1:
                    torch.distributed.barrier()
                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.save_checkpoint_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    save_checkpoint(
                        path=checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=cur_nimg,
                    )

    if dist.world_size > 1:
        torch.distributed.barrier()
    # Done.
    logger0.info("Training Completed.")

if __name__ == "__main__":
    main()