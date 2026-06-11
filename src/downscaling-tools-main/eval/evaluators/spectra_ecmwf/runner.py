"""ECMWF spectra evaluator runner — 3-stage gptosp pipeline.

Stage 1: stage prediction NetCDF files as nopoles GRIBs  (_grib_stager.py)
Stage 2: convert GRIBs to spectral harmonics via gptosp.ser (shell, AC-only)
Stage 3: compute spectra amplitudes from harmonics              (_amplitude_computer.py, needs metview)

AC-only: gptosp.ser and the eclib/pifsenv/ifs modules are not available on AG.
Resumable: stage 2 skips existing *_sh files on resubmission.
"""
from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

_WEATHER_STATE_TO_DIR = {
    "10u": "10u_sfc",
    "10v": "10v_sfc",
    "2t":  "2t_sfc",
    "sp":  "sp_sfc",
    "msl": "msl_sfc",
    "t_850": "t_850",
    "z_500": "z_500",
}

_DEFAULT_WEATHER_STATES = ["10u", "10v", "2t", "sp", "t_850", "z_500"]


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir.parent / "evaluators" / "spectra_ecmwf"

    hostname = socket.gethostname()
    if not hostname.startswith("ac"):
        raise RuntimeError(
            f"spectra_ecmwf requires gptosp.ser (AC-only). Current host: {hostname}"
        )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"spectra_ecmwf output already exists: {output_dir}")

    weather_states: list[str] = eval_config.get("weather_states", _DEFAULT_WEATHER_STATES)
    weather_states_str = ",".join(weather_states) if isinstance(weather_states, list) else str(weather_states)
    template_root: str = eval_config.get("template_root", "")
    predict_cfg = lane_config.get("predict", {})
    date_list = ",".join(predict_cfg.get("dates", []))
    step_list = ",".join(str(s) for s in predict_cfg.get("steps", [120]))

    grb_dir = output_dir / "grb"
    sh_dir  = output_dir / "spectral_harmonics"
    amp_dir = output_dir / "spectra"
    for d in (grb_dir, sh_dir, amp_dir):
        d.mkdir(parents=True, exist_ok=True)

    LOG.info("=== spectra_ecmwf 1/3: stage GRIBs ===")
    _stage_gribs(
        predictions_dir=predictions_dir,
        grb_dir=grb_dir,
        weather_states=weather_states_str,
        template_root=template_root,
        date_list=date_list,
        step_list=step_list,
        summary_path=output_dir / "staging_summary.json",
    )

    LOG.info("=== spectra_ecmwf 2/3: gptosp transforms ===")
    _run_gptosp(grb_dir=grb_dir, sh_dir=sh_dir, weather_states=weather_states)

    LOG.info("=== spectra_ecmwf 3/3: compute amplitudes ===")
    _compute_amplitudes(
        sh_dir=sh_dir,
        amp_dir=amp_dir,
        weather_states=weather_states_str,
        summary_path=output_dir / "spectra_summary.json",
    )

    return output_dir


def _stage_gribs(
    *,
    predictions_dir: Path,
    grb_dir: Path,
    weather_states: str,
    template_root: str,
    date_list: str,
    step_list: str,
    summary_path: Path,
) -> None:
    cmd = [
        sys.executable, str(_HERE / "_grib_stager.py"),
        "--predictions-dir", str(predictions_dir),
        "--out-dir", str(grb_dir),
        "--weather-states", weather_states,
        "--date-list", date_list,
        "--step-list", step_list,
        "--member-list", "ALL",
        "--summary-path", str(summary_path),
    ]
    if template_root:
        cmd += ["--template-root", template_root]
    subprocess.run(cmd, check=True)


def _run_gptosp(*, grb_dir: Path, sh_dir: Path, weather_states: list[str]) -> None:
    """Run gptosp.ser for all GRIBs; skip existing harmonics (resumable)."""
    param_dirs = [
        _WEATHER_STATE_TO_DIR[ws]
        for ws in weather_states
        if ws in _WEATHER_STATE_TO_DIR
    ]

    lines = [
        "set -euo pipefail",
        "module unload ecmwf-toolbox 2>/dev/null || true",
        "module load eclib   2>/dev/null || true",
        "module load pifsenv 2>/dev/null || true",
        "module load ifs     2>/dev/null || true",
        "export DR_HOOK_ASSERT_MPI_INITIALIZED=0",
    ]
    for pd in param_dirs:
        in_dir  = grb_dir / pd
        out_dir = sh_dir  / pd
        lines += [
            f'mkdir -p "{out_dir}"',
            f'for grb_file in "{in_dir}"/*.grb; do',
            '  [[ -f "$grb_file" ]] || continue',
            '  grb_base="$(basename "$grb_file")"',
            f'  sh_out="{out_dir}/${{grb_base}}_sh"',
            '  [[ -f "$sh_out" ]] && [[ -s "$sh_out" ]] && continue',
            '  echo "[gptosp] $grb_base"',
            '  gptosp.ser -l -g "$grb_file" -S "$sh_out"',
            'done',
        ]

    subprocess.run(
        ["bash", "--login", "-c", "\n".join(lines)],
        check=True,
    )


def _compute_amplitudes(
    *,
    sh_dir: Path,
    amp_dir: Path,
    weather_states: str,
    summary_path: Path,
) -> None:
    venv_activate = Path(sys.prefix) / "bin" / "activate"
    script = "\n".join([
        "set -euo pipefail",
        "module unload ifs         2>/dev/null || true",
        "module load ecmwf-toolbox 2>/dev/null || true",
        f'source "{venv_activate}"',
        'METVIEW_BIN="$(command -v metview 2>/dev/null || true)"',
        '[[ -n "$METVIEW_BIN" ]] && export PATH="$(dirname "$METVIEW_BIN"):$PATH"',
        f'python "{_HERE / "_amplitude_computer.py"}"'
        f' --spectral-harmonics-dir "{sh_dir}"'
        f' --out-dir "{amp_dir}"'
        f' --weather-states "{weather_states}"'
        f' --summary-path "{summary_path}"',
    ])
    subprocess.run(["bash", "--login", "-c", script], check=True)
