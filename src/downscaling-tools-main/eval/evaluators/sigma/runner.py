"""Sigma evaluator subprocess wrapper around eval.sigma_evaluator.run_sigma_evaluator.

The legacy module remains the canonical implementation. This runner translates
EvaluatorContext values into the legacy CLI argv shape.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs,
) -> Path:
    """Run sigma sweep by subprocessing into run_sigma_evaluator."""
    if not checkpoint:
        raise ValueError("sigma evaluator requires --checkpoint; pass it via eval.cli evaluate")

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "sigma"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Sigma output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(checkpoint)
    name_ckpt = ckpt_path.name
    name_exp = ckpt_path.parent.name
    ckpt_root = str(ckpt_path.parent.parent)

    out_csv = output_dir / "sigma_eval_table.csv"
    n_samples = str(eval_config.get("n_samples", 10))
    validation_frequency = str(eval_config.get("validation_frequency", "50h"))

    cmd = [
        sys.executable, "-m", "eval._backends.sigma_evaluator.run_sigma_evaluator",
        "--name_exp", name_exp,
        "--name_ckpt", name_ckpt,
        "--ckpt-root", ckpt_root,
        "--out_csv", str(out_csv),
        "--n_samples", n_samples,
        "--validation_frequency", validation_frequency,
    ]
    if eval_config.get("sigmas"):
        cmd += ["--sigmas", str(eval_config["sigmas"])]
    if eval_config.get("run_pure_noise"):
        cmd.append("--run_pure_noise")

    LOG.info("sigma subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir
