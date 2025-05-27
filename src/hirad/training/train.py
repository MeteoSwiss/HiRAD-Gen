import os
import time

import psutil
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from contextlib import nullcontext
import nvtx
import torch
from hydra.utils import to_absolute_path
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from hirad.utils.train_helpers import set_seed, configure_cuda_for_consistent_precision, \
                                        set_patch_shape, compute_num_accumulation_rounds, \
                                        is_time_for_periodic_task, handle_and_clip_gradients
from hirad.utils.checkpoint import load_checkpoint, save_checkpoint
from hirad.utils.patching import RandomPatching2D
from hirad.models import UNet, EDMPrecondSuperResolution, EDMPrecondSR
from hirad.losses import ResidualLoss, RegressionLoss, RegressionLossCE
from hirad.datasets import init_train_valid_datasets_from_config

from matplotlib import pyplot as plt

torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264  # Set to a higher value
torch._dynamo.config.verbose = True  # Enable verbose logging
torch._dynamo.config.suppress_errors = False  # Forces the error to show all details
torch._logging.set_logs(recompiles=True, graph_breaks=True)

# Define safe CUDA profiler tools that fallback to no-ops when CUDA is not available
def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()

@hydra.main(version_base=None, config_path="../conf", config_name="training")
def main(cfg: DictConfig) -> None:
    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank==0:
        writer = SummaryWriter(log_dir='tensorboard')
    logger = PythonLogger("main") # general logger
    logger0 = RankZeroLoggingWrapper(logger, dist) # rank 0 logger

    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if hasattr(cfg.dataset, "validation_path"):
        train_test_split = True
    else:
        train_test_split = False
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger0.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = os.path.join(
        cfg.training.io.get("checkpoint_dir", "."), f"checkpoints_{cfg.model.name}"
    )
    if dist.rank==0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir) # added creating checkpoint dir
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    set_seed(dist.rank)
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
    )
    logger0.info(f"Training on dataset with size {len(dataset)}")

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels


    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index() #TODO figure out what prob_channel are and update dataloader
    else:
        prob_channels = None

    # Parse the patch shape
    #TODO figure out patched diffusion and how to use it
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
    if use_patching:
        # Utility to perform patches extraction and batching
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if use_patching:
        img_in_channels += dataset_channels
    
    # Instantiate the model and move to device.
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = False
    use_apex_gn = False
    profile_mode = False

    if hasattr(cfg.training.perf, "torch_compile"):
        use_torch_compile = cfg.training.perf.torch_compile
    if hasattr(cfg.training.perf, "use_apex_gn"):
        use_apex_gn = cfg.training.perf.use_apex_gn
        model_args["use_apex_gn"] = use_apex_gn

    if hasattr(cfg.training.perf, "profile_mode"):
        profile_mode = cfg.training.perf.profile_mode
        model_args["profile_mode"] = profile_mode

    if enable_amp:
        model_args["amp_mode"] = enable_amp


    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
        model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"]
    elif cfg.model.name == "lt_aware_ce_regression":
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
        model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"] + model_args["lead_time_channels"]
    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
        model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"] + model_args["lead_time_channels"]
    else:  # diffusion or patched diffusion
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
        model_args["img_in_channels"] = img_in_channels + model_args["N_grid_channels"]
    
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

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,  # dist.find_unused_parameters,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )

    # Load the regression checkpoint if applicable #TODO test when training correction
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.isdir(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        #regression_net = torch.nn.Module() #TODO Module.from_checkpoint(regression_checkpoint_path) figure out how to save and load models, also, some basic functions like num_params, device
        #TODO make regression model loading more robust (model type is both in rergession_checkpoint_path and regression_name)
        #TODO add the option to choose epoch to load from / regression_checkpoint_path is now a folder
        regression_model_args_path = os.path.join(regression_checkpoint_path, 'model_args.json')
        if not os.path.isfile(regression_model_args_path):
            raise FileNotFoundError(f"Missing config file at '{regression_model_args_path}'.")

        with open(regression_model_args_path, 'r') as f:
            regression_model_args = json.load(f)

        regression_model_args.update({
            "use_apex_gn": use_apex_gn,
            "profile_mode": profile_mode,
            "amp_mode": enable_amp,
        })

        regression_net = UNet(**regression_model_args)

        _ = load_checkpoint(
            path=regression_checkpoint_path,
            model=regression_net,
            device=dist.device
        )
        regression_net.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        logger0.success("Loaded the pre-trained regression model")
    else:
        regression_net = None

    # Compile the model and regression net if applicable
    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)


    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    max_patch_per_gpu = getattr(cfg.training.hp, "max_patch_per_gpu", 1)

    # calculate patch per iter
    if hasattr(cfg.training.hp, "max_patch_per_gpu") and max_patch_per_gpu > 1:
        max_patch_num_per_iter = min(
            patch_num, (max_patch_per_gpu // batch_size_per_gpu)
        )  # Ensure at least 1 patch per iter
        patch_iterations = (
            patch_num + max_patch_num_per_iter - 1
        ) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]
        print(
            f"max_patch_num_per_iter is {max_patch_num_per_iter}, patch_iterations is {patch_iterations}, patch_nums_iter is {patch_nums_iter}"
        )
    else:
        patch_nums_iter = [patch_num]

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
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)

        # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=cfg.training.hp.lr, 
        betas=[0.9, 0.999], 
        eps=1e-8,
        fused=True,
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

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

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

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
                                img_clean, img_lr, *lead_time_label = next(
                                    dataset_iterator
                                )
                                if use_apex_gn:
                                    img_clean = img_clean.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                    img_lr = img_lr.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                else:
                                    img_clean = (
                                        img_clean.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                                    img_lr = (
                                        img_lr.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                            loss_fn_kwargs = {
                                "net": model,
                                "img_clean": img_clean,
                                "img_lr": img_lr,
                                "augment_pipe": None,
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

                                loss = loss.sum() / batch_size_per_gpu
                                loss_accum += loss / num_accumulation_rounds
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
                        writer.add_scalar("training_loss", average_loss, cur_nimg)
                        writer.add_scalar(
                            "training_loss_running_mean",
                            average_loss_running_mean,
                            cur_nimg,
                        )

                    ptt = is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.print_progress_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    )
                    if ptt:
                        # reset running mean of average loss
                        average_loss_running_mean = 0
                        n_average_loss_running_mean = 1

                    # Update weights.
                    with nvtx.annotate("update weights", color="blue"):

                        lr_rampup = cfg.training.hp.lr_rampup  # ramp up the learning rate
                        for g in optimizer.param_groups:
                            if lr_rampup > 0:
                                g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
                            if cur_nimg >= lr_rampup:
                                g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // 5e6)
                            current_lr = g["lr"]
                            if dist.rank == 0:
                                writer.add_scalar("learning_rate", current_lr, cur_nimg)
                        handle_and_clip_gradients(
                            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
                        )
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration

                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.print_progress_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    # Print stats if we crossed the printing threshold with this batch
                    tick_end_time = time.time()
                    fields = []
                    fields += [f"samples {cur_nimg:<9.1f}"]
                    fields += [f"training_loss {average_loss:<7.2f}"]
                    fields += [f"training_loss_running_mean {average_loss_running_mean:<7.2f}"]
                    fields += [f"learning_rate {current_lr:<7.8f}"]
                    fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
                    fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
                    fields += [
                        f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
                    ]
                    fields += [
                        f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
                    ]
                    if torch.cuda.is_available():
                        fields += [
                            f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
                        ]
                        fields += [
                            f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
                        ]
                        torch.cuda.reset_peak_memory_stats()
                    logger0.info(" ".join(fields))
                    logger0.info(img_clean.shape)
                    logger0.info(img_lr.shape)

                with nvtx.annotate("validation", color="red"):
                    # Validation
                    if validation_dataset_iterator is not None:
                        valid_loss_accum = 0
                        if is_time_for_periodic_task(
                            cur_nimg,
                            cfg.training.io.validation_freq,
                            done,
                            cfg.training.hp.total_batch_size,
                            dist.rank,
                        ):
                            with torch.no_grad():
                                for _ in range(cfg.training.io.validation_steps):
                                    (
                                        img_clean_valid,
                                        img_lr_valid,
                                        *lead_time_label_valid,
                                    ) = next(validation_dataset_iterator)

                                    if use_apex_gn:
                                        img_clean_valid = img_clean_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)
                                        img_lr_valid = img_lr_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)

                                    else:
                                        img_clean_valid = (
                                            img_clean_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )
                                        img_lr_valid = (
                                            img_lr_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )

                                    loss_valid_kwargs = {
                                        "net": model,
                                        "img_clean": img_clean_valid,
                                        "img_lr": img_lr_valid,
                                        "augment_pipe": None,
                                    }
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs[
                                            "use_patch_grad_acc"
                                        ] = use_patch_grad_acc
                                    if lead_time_label_valid:
                                        lead_time_label_valid = (
                                            lead_time_label_valid[0]
                                            .to(dist.device)
                                            .contiguous()
                                        )
                                        loss_valid_kwargs.update(
                                            {"lead_time_label": lead_time_label_valid}
                                        )
                                    if use_patch_grad_acc:
                                        loss_fn.y_mean = None

                                    for patch_num_per_iter in patch_nums_iter:
                                        if patching is not None:
                                            patching.set_patch_num(patch_num_per_iter)
                                            loss_fn_kwargs.update(
                                                {"patching": patching}
                                            )
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            loss_valid = loss_fn(**loss_valid_kwargs)

                                        loss_valid = (
                                            (loss_valid.sum() / batch_size_per_gpu)
                                            .cpu()
                                            .item()
                                        )
                                        valid_loss_accum += (
                                            loss_valid
                                            / cfg.training.io.validation_steps
                                        )
                                valid_loss_sum = torch.tensor(
                                    [valid_loss_accum], device=dist.device
                                )
                                if dist.world_size > 1:
                                    torch.distributed.barrier()
                                    torch.distributed.all_reduce(
                                        valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                                    )
                                average_valid_loss = valid_loss_sum / dist.world_size
                                if dist.rank == 0:
                                    writer.add_scalar(
                                        "validation_loss", average_valid_loss, cur_nimg
                                    )


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

    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()