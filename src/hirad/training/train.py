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
from torchinfo import summary

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from hirad.utils.train_helpers import set_seed, configure_cuda_for_consistent_precision, \
                                        set_patch_shape, compute_num_accumulation_rounds, calculate_patch_per_iter, \
                                        is_time_for_periodic_task, handle_and_clip_gradients, \
                                        init_mlflow, update_learning_rate, log_training_progress, \
                                        cuda_profiler, cuda_profiler_start, cuda_profiler_stop, profiler_emit_nvtx
from hirad.utils.checkpoint import load_checkpoint, save_checkpoint
from hirad.utils.patching import RandomPatching2D
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon
from hirad.models import UNet
from hirad.losses import ResidualLoss, RegressionLoss, DiffusionLoss
from hirad.datasets import init_train_valid_datasets_from_config, get_dataset_and_sampler_inference
from hirad.training.training_manager import TrainingManagerCorrDiff, TrainingManagerDiT



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
    use_apex_gn = getattr(cfg.training.perf, "use_apex_gn", True)
    profile_mode = getattr(cfg.training.perf, "profile_mode", False)
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = getattr(cfg.training.perf, "songunet_checkpoint_level", None)
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
    is_real_target = dataset_cfg.get("type").split("_")[-1] == "real"
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
    training_manager_args = {
        "dist": dist,
        "logger": logger0,
        "dataset": dataset,
        "input_dtype": input_dtype,
        "img_shape": img_shape,
        "n_month_hour_channels": n_month_hour_channels,
        "fp16": fp16,
        "enable_amp": enable_amp,
        "amp_dtype": amp_dtype,
        "is_real_target": is_real_target,
        "logging_method": cfg.logging.get("method", None),
        "use_apex_gn": use_apex_gn,
    }
    if cfg.model.name in {"diffusion_transformer"}:
        training_manager_args["n_prev_hr_frames"] = dataset_cfg.get("n_prev_hr_frames", 0)
        training_manager_args["prev_hr_dropout"] = cfg.training.hp.get("prev_hr_dropout", 0.0)
        training_manager = TrainingManagerDiT(**training_manager_args)
    else:
        training_manager = TrainingManagerCorrDiff(
                                            **training_manager_args,
                                            profile_mode=profile_mode, 
                                            songunet_checkpoint_level=songunet_checkpoint_level,
                                            use_patching=use_patching,
                                            hr_mean_conditioning=cfg.model.get("hr_mean_conditioning", False),
                                            )
    

    # Create the model and move it to the appropriate device and memory format based on the optimization configuration
    model, model_args = training_manager.create_model(cfg.model.name, cfg.model.get("model_args", None))

    logger0.info(f"Model attention backend: {model.model.attn_kwargs_forward}")

    # Print the model summary
    if dist.rank == 0:
        summary(model, input_size=[(1, 4, *img_shape), (1, 13+1, *img_shape), (1,1)], device=dist.device)

    # raise NotImplementedError("Check if model_args are correct when using patching - img_in_channels should include global channels and lead time channels if applicable")

    
    model.train().requires_grad_(True).to(dist.device)

    if dist.rank==0 and not os.path.exists(os.path.join(checkpoint_dir, 'model_args.json')):
        with open(os.path.join(checkpoint_dir, f'model_args.json'), 'w') as f:
            # json.dump(OmegaConf.to_container(model_args, resolve=True) if isinstance(model_args, DictConfig) else model_args, f)
            json.dump(OmegaConf.to_container(OmegaConf.structured(model_args), resolve=True), f)
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
    elif cfg.model.name == "diffusion_transformer":
        loss_fn = DiffusionLoss()

    # Instantiate the optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=cfg.training.hp.lr, 
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01,
        fused=True,
    )

    # Set up the learning rate scheduler with linear warmup and cosine annealing
    total_steps = cfg.training.hp.training_duration // cfg.training.hp.total_batch_size
    warmup_steps = max(1, cfg.training.hp.lr_rampup // cfg.training.hp.total_batch_size)
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8 / cfg.training.hp.lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=cfg.training.hp.get("lr_min", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Load optimizer checkpoint if it exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dist.device,
        )
    except:
        cur_nimg = 0

    # Fast-forward scheduler to current step when resuming from checkpoint
    # only needed if no scheduler state was saved previously
    current_step = cur_nimg // cfg.training.hp.total_batch_size
    if current_step > 0 and scheduler.last_epoch == 0:
        logger0.info(f"No scheduler state found, fast-forwarding LR scheduler to step {current_step}")
        for _ in range(current_step):
            scheduler.step()

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

                                # loss = loss.sum() / batch_size_per_gpu / patch_num_per_iter / img_shape[0] / img_shape[1] / len(dataset.output_channels())
                                loss = loss.mean()
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

                    

                    if dist.rank == 0:
                        # 1. Calculate the total L2 norm of all parameters
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

                    # Update weights.
                    with nvtx.annotate("update weights", color="blue"):
                        handle_and_clip_gradients(
                            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
                        )

                    if dist.rank == 0:
                        # 1. Calculate the total L2 norm of all parameters
                        total_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']

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
                                                # 2. Log to MLflow
                        mlflow.log_metric("grad_norm", total_norm.item(), step=cur_nimg+cfg.training.hp.total_batch_size)
                        mlflow.log_metric("grad_norm_clipped", total_norm_clipped.item(), step=cur_nimg+cfg.training.hp.total_batch_size)


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
                        scheduler=scheduler,
                        epoch=cur_nimg,
                    )

    if dist.world_size > 1:
        torch.distributed.barrier()
    # Done.
    logger0.info("Training Completed.")

if __name__ == "__main__":
    main()