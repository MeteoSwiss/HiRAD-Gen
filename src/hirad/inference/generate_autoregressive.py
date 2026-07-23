"""Autoregressive DiT inference script.

Generates a time sequence where each step conditions the DiT on the ensemble mean
of the previous step's output (in normalized space).  The first step is always
unconditional: a zero tensor (= the null-conditioning token used during training)
is supplied as the previous HR.

Supports both checkpoints trained with temporal conditioning
(n_prev_hr_frames > 0 in model_args.json) and legacy checkpoints trained without
it (n_prev_hr_frames == 0).  In the legacy case the script falls back to standard
stateless generation identical to generate.py.

Usage
-----
python generate_autoregressive.py --config-name=generate generation=ardit
"""
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

from hirad.models import EDMPrecondSuperResolution
from hirad.inference import GeneratorDiT
from hirad.utils.inference_utils import save_results_as_torch
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape


# ---------------------------------------------------------------------------
# Helpers (identical to generate.py)
# ---------------------------------------------------------------------------

def _log_model_summary(label: str, model: torch.nn.Module, logger, max_depth: int = 2) -> None:
    """Log a concise module-tree summary with per-node parameter counts."""
    def _param_count(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    def _walk(m: torch.nn.Module, depth: int) -> list[str]:
        lines = []
        indent = "  " * (depth + 1)
        for name, p in m.named_parameters(recurse=False):
            lines.append(f"{indent}[{name}] nn.Parameter{' ' * 28} {p.numel():>15,} params  shape={list(p.shape)}")
        for name, child in m.named_children():
            n = _param_count(child)
            lines.append(f"{indent}[{name}] {type(child).__name__:<40s} {n:>15,} params")
            if depth < max_depth - 1:
                lines.extend(_walk(child, depth + 1))
        return lines

    total = _param_count(model)
    lines = [f"{label} ({type(model).__name__}): {total:,} params"] + _walk(model, 0)
    for line in lines:
        logger.info(line)


def _sync_t() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate_autoregressive")
def main(cfg: DictConfig) -> None:
    """Autoregressive DiT generation over a time range.

    For the first time step the model is called with a zero previous-HR tensor
    (unconditional).  For every subsequent step the ensemble mean from the
    previous step (still in normalized space) is passed as prev_hr.
    """
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate_autoregressive")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    if dist.world_size > 1:
        torch.distributed.barrier()

    input_dtype = torch.float16 if cfg.generation.perf.get("force_fp16", False) else torch.float32

    # ------------------------------------------------------------------
    # Parse time range
    # ------------------------------------------------------------------
    if cfg.generation.get("times_range", None) and cfg.generation.get("times", None):
        raise ValueError("Provide either times_range or times, not both.")
    if cfg.generation.get("times_range", None):
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M")
    elif cfg.generation.get("times", None):
        times = cfg.generation.times
    else:
        raise ValueError("Either times_range or times must be provided.")

    logger0.info(f"Generating {len(times)} time steps: {times[0]} → {times[-1]}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    has_lead_time = cfg.generation.get("has_lead_time", False)
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

    # ------------------------------------------------------------------
    # Load DiT checkpoint
    # ------------------------------------------------------------------
    dit_ckpt_path = cfg.generation.io.dit_ckpt_path
    logger0.info(f'Loading DiT model from "{dit_ckpt_path}"...')

    dit_model_args_path = os.path.join(dit_ckpt_path, "model_args.json")
    if not os.path.isfile(dit_model_args_path):
        raise FileNotFoundError(f"Missing model_args.json at '{dit_model_args_path}'.")
    with open(dit_model_args_path, "r") as f:
        dit_model_args = json.load(f)

    # Disable AMP at inference time
    if "amp_mode" in dit_model_args:
        dit_model_args["amp_mode"] = False

    # Detect whether the checkpoint was trained with temporal HR conditioning.
    # The training manager adds n_prev_hr_frames * n_out_channels to img_in_channels,
    # but does NOT store n_prev_hr_frames explicitly in model_args.json.
    # We store it in the generation config instead so the user controls it.
    n_prev_hr_frames = cfg.generation.get("n_prev_hr_frames", 0)
    # How the previous-HR condition is sourced each step (diagnostic switch):
    #   "self"   – the model's own previous prediction (default; true AR rollout)
    #   "zeros"  – always unconditional (drop the prev-HR condition every step);
    #              isolates whether the AR condition causes the diurnal damping
    #   "target" – teacher-force with the TRUE previous-HR target; the gap vs
    #              "self" quantifies exposure bias (train/inference mismatch)
    prev_hr_source = cfg.generation.get("prev_hr_source", "self")
    if n_prev_hr_frames > 0 and prev_hr_source not in ("self", "zeros", "target"):
        raise ValueError(
            f"generation.prev_hr_source must be one of 'self', 'zeros', 'target'; "
            f"got '{prev_hr_source}'."
        )
    if n_prev_hr_frames > 0:
        logger0.info(
            f"Autoregressive mode: conditioning on previous {n_prev_hr_frames} HR frame(s) "
            f"[prev_hr_source='{prev_hr_source}']. First step is always zeros (unconditional)."
        )
    else:
        logger0.info("Standard (non-autoregressive) mode: no previous-HR conditioning.")

    use_apex_gn = True  # DiT always uses channels_last

    net_dit = EDMPrecondSuperResolution(**dit_model_args)
    _ = load_checkpoint(path=dit_ckpt_path, model=net_dit, device=dist.device)
    net_dit = net_dit.eval().to(device)
    if use_apex_gn:
        net_dit = net_dit.to(memory_format=torch.channels_last)
    if cfg.generation.perf.force_fp16:
        net_dit.use_fp16 = True
    _log_model_summary("DiT network", net_dit, logger0)

    # ------------------------------------------------------------------
    # Torch compile
    # ------------------------------------------------------------------
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        net_dit = torch.compile(net_dit)

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------
    generator = GeneratorDiT(
        model=net_dit,
        batch_size=cfg.generation.seed_batch_size,
        ensemble_size=cfg.generation.num_ensembles,
        n_out_channels=img_out_channels,
        dist=dist,
    )

    # Patching is not supported for autoregressive runs: the state between
    # patches would need separate bookkeeping.  Use full-domain inference.
    if cfg.generation.get("patching", False):
        raise ValueError(
            "Patch-based inference is not supported in autoregressive mode. "
            "Set patching: False in the generation config."
        )

    sampler_params = dict(OmegaConf.to_container(cfg.sampler.params, resolve=True)) if "params" in cfg.sampler else {}
    sampler_params["use_apex_gn"] = use_apex_gn
    generator.initialize_sampler(cfg.sampler.type, **sampler_params)

    # ------------------------------------------------------------------
    # Static channels (unchanged from generate.py)
    # ------------------------------------------------------------------
    static_channels = dataset.get_static_data()
    if static_channels is not None:
        static_channels = static_channels[None, ::].flip(-2)
        if use_apex_gn:
            static_channels = static_channels.to(
                dist.device, dtype=input_dtype, non_blocking=True
            ).to(memory_format=torch.channels_last)
        else:
            static_channels = static_channels.to(dist.device).to(input_dtype).contiguous()

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    output_path = getattr(cfg.generation.io, "output_path", "./outputs")
    logger0.info(f"Saving results to {output_path}")

    warmup_steps = min(len(times) - 1, 2)
    enable_timing = cfg.generation.perf.get("enable_timing", True)
    _t = _sync_t if enable_timing else (lambda: 0.0)

    use_cuda_timing = torch.cuda.is_available()
    if use_cuda_timing:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
    else:
        class _DummyEvent:
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self, _): return 0
        start_evt = end_evt = _DummyEvent()

    step_timings = defaultdict(float)
    timed_step_count = 0

    # Previous-HR state, in normalized space.
    #   Step 0 (first): zeros of shape (1, C_hr, H, W) — same unconditional
    #                   token for all members, identical to training dropout.
    #   Step t > 0:     full ensemble of shape (N, C_hr, H, W) — member i
    #                   conditions on its own output from the previous step.
    #
    # Using per-member conditioning requires randomize=False so that
    # rank_batches (and therefore member ordering) is stable across steps.
    prev_hr: torch.Tensor | None = None
    if n_prev_hr_frames > 0:
        # if cfg.generation.get("randomize", False):
        #     logger0.info(
        #         "WARNING: randomize=True with per-member autoregressive conditioning "
        #         "is not supported — member ordering is undefined across steps. "
        #         "Set randomize: False for consistent trajectories."
        #     )
        prev_hr = torch.zeros(
            1, img_out_channels, *img_shape, device=device, dtype=input_dtype
        )
        logger0.info(
            f"Initialized prev_hr to zeros (unconditional first step)  "
            f"shape={list(prev_hr.shape)}"
        )

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if use_cuda_timing and cfg.generation.perf.get("profile", False)
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if use_cuda_timing and cfg.generation.perf.get("profile", False)
        else contextlib.nullcontext()
    )

    with torch_cuda_profiler:
        with torch_nvtx_profiler:
            with torch.inference_mode():

                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=1,
                    pin_memory=True,
                    num_workers=4,
                    persistent_workers=True,
                )

                if dist.rank == 0:
                    writer_executor = ThreadPoolExecutor(
                        max_workers=cfg.generation.perf.num_writer_workers
                    )
                    writer_threads = []

                all_times = dataset.time()
                time_index = -1
                t_iter_end = _t()

                for index, (image_tar, image_lr, *date_str) in enumerate(iter(data_loader)):
                    t_iter_start = _t()
                    t_data_load = t_iter_start - t_iter_end

                    t_preproc_start = _t()
                    time_index += 1
                    current_time = all_times[sampler[time_index]]
                    is_first_step = (time_index == 0)

                    if dist.rank == 0:
                        conditioning_desc = (
                            "unconditional (zeros, all members share same prev_hr)"
                            if (n_prev_hr_frames > 0 and is_first_step)
                            else ("per-member prev_hr" if n_prev_hr_frames > 0 else "no prev_hr")
                        )
                        logger0.info(
                            f"Step {time_index:4d}  time={current_time}  [{conditioning_desc}]"
                        )

                    if time_index == warmup_steps:
                        start_evt.record()
                        if use_cuda_timing:
                            torch.cuda.reset_peak_memory_stats()

                    savedir = os.path.join(output_path, current_time)
                    os.makedirs(savedir, exist_ok=True)

                    # ---- Preprocess ground-truth target (for saving) ----
                    if is_real_target:
                        image_tar = regrid_icon_to_rotlatlon(
                            image_tar.to(dist.device, dtype=input_dtype),
                            dataset.regrid_indices_real,
                            dataset.regrid_weights_real,
                        )
                        if dataset.trim_edge > 0:
                            image_tar = image_tar[
                                :, :,
                                dataset.trim_edge:-dataset.trim_edge,
                                dataset.trim_edge:-dataset.trim_edge,
                            ]
                    else:
                        image_tar = image_tar.reshape(
                            *image_tar.shape[:-1], *dataset.image_shape()
                        )

                    # ---- Preprocess low-res ERA5 input ----
                    image_lr = (
                        dataset.interpolator(image_lr.to(dist.device, dtype=input_dtype))
                        .reshape(*image_lr.shape[:-1], *dataset.image_shape())
                        .flip(-2)
                    )
                    image_lr = dataset.normalize_input(image_lr)
                    if use_apex_gn:
                        image_lr = image_lr.to(memory_format=torch.channels_last)

                    # ---- Date embedding ----
                    date_embedding = None
                    if dataset._n_month_hour_channels:
                        date_embedding = dataset.make_time_grids(
                            *date_str, dist.device, dtype=input_dtype
                        )

                    random_seed = (
                        cfg.generation.get("random_seed", None) + index
                        if cfg.generation.get("randomize", False)
                        and cfg.generation.get("random_seed", None) is not None
                        else None
                    )
                    t_preproc_end = _t()

                    # ---- Generate ensemble ----
                    t_gen_start = _t()
                    image_out, _ = generator.generate(
                        image_lr,
                        prev_hr=prev_hr,          # None for legacy ckpt; zeros/mean for new ckpt
                        static_channels=static_channels,
                        date_embedding=date_embedding,
                        lead_time_label=None,
                        randomize=cfg.generation.get("randomize", False),
                        random_seed=random_seed,
                        use_apex_gn=use_apex_gn,
                        skip_timing=(not enable_timing or time_index < warmup_steps),
                    )
                    t_gen_end = _t()

                    # ---- Update prev_hr for the next step ----
                    if n_prev_hr_frames > 0:
                        if prev_hr_source == "self":
                            # True AR rollout: each rank already produced its own
                            # output slice BEFORE the gather (stored as
                            # generator._last_local_output). Use it directly as the
                            # per-member prev_hr — no inter-rank broadcast needed.
                            prev_hr = generator._last_local_output
                        elif prev_hr_source == "target":
                            # Teacher-forcing diagnostic: condition the next step on
                            # the TRUE current-step target, preprocessed exactly as
                            # in training (training_manager.load_and_preprocess_batch):
                            # normalize_output of the flipped regridded/trimmed target.
                            # Shape (1, C, H, W) → shared across all members.
                            prev_hr_tf = dataset.normalize_output(
                                image_tar.to(dist.device, dtype=input_dtype).flip(-2)
                            )
                            if use_apex_gn:
                                prev_hr_tf = prev_hr_tf.to(memory_format=torch.channels_last)
                            else:
                                prev_hr_tf = prev_hr_tf.contiguous()
                            prev_hr = prev_hr_tf
                        elif prev_hr_source == "zeros":
                            # Unconditional every step: leave prev_hr as the initial
                            # zeros so the AR condition is never supplied.
                            pass

                    # ---- Post-process and save (rank 0 only) ----
                    t_postproc_start = _t()
                    if dist.rank == 0:
                        image_tar_np = image_tar[0].squeeze().cpu().numpy()
                        prediction_ensemble = (
                            dataset.denormalize_output(image_out).squeeze().flip(-2).cpu().numpy()
                        )
                        baseline = (
                            dataset.denormalize_input(image_lr)[0].squeeze().flip(-2).cpu().numpy()
                        )
                    t_postproc_end = _t()

                    t_write_start = _t()
                    if dist.rank == 0:
                        writer_threads.append(
                            writer_executor.submit(
                                save_results_as_torch,
                                savedir,
                                current_time,
                                prediction_ensemble,
                                image_tar_np,
                                baseline,
                                None,  # no regression mean for DiT
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

                end_evt.record()
                end_evt.synchronize()
                elapsed_time = (
                    start_evt.elapsed_time(end_evt) / 1000.0 if use_cuda_timing else 0.0
                )
                timed_steps = time_index + 1 - warmup_steps
                if dist.rank == 0 and use_cuda_timing:
                    avg_per_elem = elapsed_time / max(timed_steps, 1)
                    logger0.info(
                        f"Total time for {timed_steps} timed steps = {elapsed_time:.2f} s  "
                        f"({avg_per_elem:.3f} s/step)"
                    )

                if dist.rank == 0 and timed_step_count > 0:
                    logger0.info(
                        "--- Timing breakdown (avg over timed steps, wall-clock GPU-synced) ---"
                    )
                    for key in [
                        "data_loading", "preprocessing", "generation",
                        "postprocessing", "io_submit",
                    ]:
                        avg = step_timings[key] / timed_step_count
                        logger0.info(f"  {key:20s}: {avg:.3f} s/step")
                    gen_timings = generator.get_timings()
                    if gen_timings:
                        logger0.info(
                            "--- Generator internal timing (avg, warmup excluded) ---"
                        )
                        for key, (total, count) in gen_timings.items():
                            avg = total / count if count > 0 else 0.0
                            if key.endswith("_calls"):
                                logger0.info(f"  {key:20s}: {avg:.1f} calls/step")
                            else:
                                logger0.info(f"  {key:20s}: {avg:.3f} s/step")

                if dist.rank == 0 and use_cuda_timing:
                    _gib = 1024 ** 3
                    logger0.info("--- Peak GPU memory (timed steps only) ---")
                    logger0.info(
                        f"  allocated: {torch.cuda.max_memory_allocated() / _gib:.2f} GiB"
                    )
                    logger0.info(
                        f"  reserved : {torch.cuda.max_memory_reserved()  / _gib:.2f} GiB"
                    )

                if dist.rank == 0:
                    for thread in list(writer_threads):
                        thread.result()
                        writer_threads.remove(thread)
                    writer_executor.shutdown()

    logger0.info("Autoregressive generation completed.")


if __name__ == "__main__":
    main()
