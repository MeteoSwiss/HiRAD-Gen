#!/usr/bin/env python3
"""Validation checkpoint checker for the anemoi-core downscaling validation suite.

Loads a training checkpoint, feeds the same data used during training through the
model, and computes RMSE to verify overfitting or convergence.

Supports two modes:
  - deterministic: single forward pass at σ=500000
  - diffusion: Karras sampling with configurable σ_max, num_steps

Usage:
  python validate_checkpoint.py \
    --checkpoint /path/to/last.ckpt \
    --mode deterministic \
    --rmse-threshold 1e-4

  python validate_checkpoint.py \
    --checkpoint /path/to/last.ckpt \
    --mode diffusion \
    --sigma-max 100 \
    --num-steps 20 \
    --rmse-threshold 0.1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_checkpoint_and_data(checkpoint_path: str):
    """Load a training checkpoint and reconstruct model + datamodule."""
    from anemoi.training.train.train import AnemoiTrainer

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["hyper_parameters"]["config"]

    trainer = AnemoiTrainer(config)
    datamodule = trainer.datamodule

    task_cls_path = config.training.model_task
    module_path, class_name = task_cls_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    TaskClass = getattr(mod, class_name)

    task = TaskClass(
        config=config,
        graph_data=trainer.graph_data,
        truncation_data=trainer.truncation_data,
        statistics=datamodule.statistics,
        statistics_tendencies=getattr(datamodule, "statistics_tendencies", None),
        data_indices=trainer.data_indices,
        metadata=datamodule.metadata,
        supporting_arrays=getattr(datamodule, "supporting_arrays", {}),
    )
    task.load_state_dict(ckpt["state_dict"], strict=False)
    task.eval()

    return task, datamodule, config


def get_first_batch(datamodule):
    """Get the first training batch."""
    datamodule.setup(stage="fit")
    train_dl = datamodule.train_dataloader()
    batch = next(iter(train_dl))
    return batch


def validate_deterministic(task, batch, device):
    """Single forward pass at σ=500000 (deterministic training sigma)."""
    task = task.to(device)
    batch = [b.to(device) for b in batch]

    with torch.no_grad():
        loss, metrics, y_preds = task._step(batch, batch_idx=0, training_mode=False)

    y_pred_full = y_preds[0]  # full state prediction (residual + interp)
    y_pred_residual = y_preds[1]  # residual prediction

    x_in, x_in_hres, y = batch

    # Compute RMSE on the full state prediction vs ground truth
    y_target = y[:, :, :, :, task.data_indices.data.output.full]
    rmse = torch.sqrt(torch.mean((y_pred_full - y_target) ** 2)).item()

    # Per-variable RMSE
    var_names = list(task.data_indices.data.output.name_to_index.keys())
    per_var_rmse = {}
    for name, idx in task.data_indices.data.output.name_to_index.items():
        if idx in task.data_indices.data.output.full:
            pos = list(task.data_indices.data.output.full).index(idx)
            var_rmse = torch.sqrt(torch.mean((y_pred_full[..., pos] - y_target[..., pos]) ** 2)).item()
            per_var_rmse[name] = var_rmse

    return rmse, per_var_rmse, loss.item()


def validate_diffusion(task, batch, device, sigma_max=100, num_steps=20):
    """Karras sampling from noise."""
    task = task.to(device)
    batch = [b.to(device) for b in batch]

    x_in, x_in_hres, y = batch

    noise_scheduler_params = {
        "schedule_type": "karras",
        "sigma_max": sigma_max,
        "sigma_min": 0.03,
        "rho": 7.0,
        "num_steps": num_steps,
    }
    sampler_params = {
        "sampler": "heun",
        "S_churn": 0,
        "S_min": 0.0,
        "S_max": float("inf"),
        "S_noise": 1.0,
    }

    with torch.no_grad():
        out = task.model.model.predict_step(
            x_in_lres=x_in,
            x_in_hres=x_in_hres,
            pre_processors=task.model.pre_processors,
            post_processors=task.model.post_processors,
            multi_step=1,
            model_comm_group=None,
            gather_out=True,
            noise_scheduler_params=noise_scheduler_params,
            sampler_params=sampler_params,
        )

    y_target = y[..., task.data_indices.data.output.full]
    if out.shape != y_target.shape:
        LOGGER.warning("Shape mismatch: out=%s, target=%s", out.shape, y_target.shape)
        min_vars = min(out.shape[-1], y_target.shape[-1])
        out = out[..., :min_vars]
        y_target = y_target[..., :min_vars]

    rmse = torch.sqrt(torch.mean((out - y_target) ** 2)).item()

    per_var_rmse = {}
    for name, idx in task.data_indices.data.output.name_to_index.items():
        if idx in task.data_indices.data.output.full:
            pos = list(task.data_indices.data.output.full).index(idx)
            if pos < out.shape[-1]:
                var_rmse = torch.sqrt(torch.mean((out[..., pos] - y_target[..., pos]) ** 2)).item()
                per_var_rmse[name] = var_rmse

    return rmse, per_var_rmse


def main():
    parser = argparse.ArgumentParser(description="Validate an anemoi-core training checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint (.ckpt)")
    parser.add_argument("--mode", choices=["deterministic", "diffusion"], required=True)
    parser.add_argument("--rmse-threshold", type=float, required=True, help="RMSE threshold for PASS/FAIL")
    parser.add_argument("--sigma-max", type=float, default=100, help="Karras sigma_max (diffusion mode)")
    parser.add_argument("--num-steps", type=int, default=20, help="Karras num_steps (diffusion mode)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try to find last.ckpt in the checkpoint directory
        candidates = list(ckpt_path.parent.glob("last.ckpt")) if ckpt_path.parent.exists() else []
        if candidates:
            ckpt_path = candidates[0]
        else:
            LOGGER.error("Checkpoint not found: %s", args.checkpoint)
            sys.exit(1)

    LOGGER.info("Loading checkpoint: %s", ckpt_path)
    task, datamodule, config = load_checkpoint_and_data(str(ckpt_path))

    LOGGER.info("Loading first training batch...")
    batch = get_first_batch(datamodule)

    if args.mode == "deterministic":
        LOGGER.info("Running deterministic validation (σ=500000)...")
        rmse, per_var_rmse, loss = validate_deterministic(task, batch, args.device)
        LOGGER.info("Training loss at checkpoint: %.6f", loss)
    else:
        LOGGER.info("Running diffusion validation (Karras σ_max=%s, %d steps)...", args.sigma_max, args.num_steps)
        rmse, per_var_rmse = validate_diffusion(
            task, batch, args.device,
            sigma_max=args.sigma_max,
            num_steps=args.num_steps,
        )

    LOGGER.info("=" * 60)
    LOGGER.info("Overall RMSE: %.6f", rmse)
    LOGGER.info("Per-variable RMSE:")
    for name, val in sorted(per_var_rmse.items()):
        LOGGER.info("  %-20s  %.6f", name, val)
    LOGGER.info("Threshold: %.6f", args.rmse_threshold)

    if rmse < args.rmse_threshold:
        LOGGER.info("RESULT: ✅ PASS (RMSE %.6f < %.6f)", rmse, args.rmse_threshold)
        sys.exit(0)
    else:
        LOGGER.info("RESULT: ❌ FAIL (RMSE %.6f >= %.6f)", rmse, args.rmse_threshold)
        sys.exit(1)


if __name__ == "__main__":
    main()
