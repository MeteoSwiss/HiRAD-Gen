"""Spectra (proxy) evaluator — subprocess wrapper around proxy_runner.py."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

_SPECTRA_SCRIPT = Path(__file__).resolve().parent / "proxy_runner.py"


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    **kwargs,
) -> Path:
    """Run proxy spectra computation via proxy_runner.py."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "spectra"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Spectra output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_states = eval_config.get("weather_states", "10u,10v,2t,msl,t_850,z_500")
    nside = str(eval_config.get("nside", 64))
    lmax = str(eval_config.get("lmax", 319))
    member_aggregation = eval_config.get("member_aggregation", "per-file-mean")

    cmd = [
        sys.executable, str(_SPECTRA_SCRIPT),
        "--predictions-dir", str(predictions_dir),
        "--out-dir", str(output_dir),
        "--run-label", run_label or predictions_dir.name,
        "--weather-states", weather_states,
        "--nside", nside,
        "--lmax", lmax,
        "--member-aggregation", member_aggregation,
        "--consolidated-pdf", str(output_dir / "all_spectra_proxy.pdf"),
    ]

    LOG.info("spectra subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir
