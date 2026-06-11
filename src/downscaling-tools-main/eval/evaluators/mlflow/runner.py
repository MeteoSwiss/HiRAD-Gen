"""MLflow training-loss evaluator runner.

Subprocesses _import.py which handles:
  - Phase A: extract run UUID from checkpoint path
  - Phase B: locate MLflow logs (ATOS local or Jupiter rsync)
  - Phase C: load metrics, generate key_vars / overview / all_vars / loss_curves plots

Soft-fails: if no matching MLflow run is found, logs a warning and returns
without raising (consistent with the best-effort nature of this evaluator).
Requires: checkpoint path passed via --checkpoint to eval.cli.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs,
) -> Path | None:
    if not checkpoint:
        LOG.warning(
            "mlflow evaluator requires a checkpoint path (pass --checkpoint). Skipping."
        )
        return None

    output_dir = Path(output_dir) if output_dir else Path(predictions_dir).parent / "evaluators" / "mlflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(_HERE / "_import.py"),
        "--checkpoint-path", str(checkpoint),
        "--run-root", str(output_dir),
    ]
    if overwrite:
        cmd.append("--force")

    LOG.info("mlflow evaluator: running _import.py for checkpoint %s", checkpoint)
    result = subprocess.run(cmd, check=False)
    if result.returncode not in (0, 2):
        LOG.warning(
            "mlflow _import.py exited with code %d (0=ok, 2=no logs found)",
            result.returncode,
        )

    return output_dir
