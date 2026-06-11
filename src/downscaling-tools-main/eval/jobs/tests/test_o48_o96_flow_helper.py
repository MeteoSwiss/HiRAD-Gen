from __future__ import annotations

import ast
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "eval/jobs/templates/submit_o48_o96_manual_eval_flow.sh"
FINALIZE_TEMPLATE = ROOT / "eval/jobs/templates/finalize_lean_eval_layout.sbatch"


def _extract_shell_string(script_text: str, var: str) -> str:
    match = re.search(rf"^{re.escape(var)}=(.+)$", script_text, re.MULTILINE)
    assert match is not None, f"{var} not found in rendered script"
    return ast.literal_eval(match.group(1))


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_fake_checkpoint(path: Path, *, lres: int, hres: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hyper_parameters": {
            "config": {
                "dataloader": {
                    "validation": {
                        "dataset": {
                            "zip": [
                                {"name": "lres", "dataset": f"/tmp/data/o{lres}/input.zarr"},
                                {"name": "hres", "dataset": f"/tmp/data/o{hres}/truth.zarr"},
                            ]
                        }
                    }
                },
                "multi_dataset_normalizer": True,
            }
        }
    }
    torch.save(payload, path)
    return path


def _make_fake_hostname(path: Path, host_short: str) -> Path:
    script = textwrap.dedent(
        f"""\
        #!/bin/sh
        if [ "$1" = "-s" ]; then
          printf '%s\\n' {host_short!r}
        else
          printf '%s\\n' {host_short!r}
        fi
        """
    )
    return _write_executable(path, script)


def _touch_required_source_gribs(root: Path, dates: list[str]) -> None:
    for date in dates:
        for name in (
            f"enfo_o48_0001_date{date}_time0000_mem1to10_step24to120_sfc.grib",
            f"enfo_o48_0001_date{date}_time0000_mem1to10_step24to120_pl.grib",
            f"enfo_o96_0001_date{date}_time0000_step24to120_sfc.grib",
            f"iekm_o96_iekm_date{date}_time0000_step24to120_sfc_y.grib",
            f"iekm_o96_iekm_date{date}_time0000_step24to120_pl_y.grib",
        ):
            (root / name).write_text("stub\n", encoding="utf-8")


def _run_helper(
    tmp_path: Path,
    *,
    phase: str,
    run_id_override: str = "",
    extra_env: dict[str, str] | None = None,
    host_family: str = "ag",
    host_short: str = "ag6-test",
    populate_source_gribs: bool = True,
) -> subprocess.CompletedProcess[str]:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    submit_root = tmp_path / "submit"
    output_root = tmp_path / "output"
    submit_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    fake_checkpoint = _write_fake_checkpoint(
        tmp_path / "checkpoints" / "95a07500d80247dbbd4b143d78db805d" / "anemoi-by_step-epoch_013-step_100000.ckpt",
        lres=48,
        hres=96,
    )
    _make_fake_hostname(bin_dir / "hostname", host_short)

    if populate_source_gribs:
        _touch_required_source_gribs(source_root, ["20250926", "20250927", "20250928", "20250929", "20250930"])

    if run_id_override:
        (output_root / run_id_override).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_PATH": str(fake_checkpoint),
            "SOURCE_HPC": host_family,
            "SOURCE_GRIB_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "SUBMIT_ROOT": str(submit_root),
            "PROFILE_PYTHON": sys.executable,
            "NO_SUBMIT": "1",
            "ALLOW_OVERWRITE": "1",
            "RUN_DATE_UTC": "20260330",
            "RUN_SUFFIX": "manual_eval",
            "PHASE": phase,
        }
    )
    if run_id_override:
        env["RUN_ID_OVERRIDE"] = run_id_override
    if extra_env:
        env.update(extra_env)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(HELPER)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_o48_o96_helper_proxy_render_uses_truth_bundle_stage(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy", extra_env={"RUN_STORM_PLOTS": "1"})
    assert result.returncode == 0, result.stderr

    run_id = "manual_95a07500_new_o48_o96_20260330_manual_eval"
    submit_dir = tmp_path / "submit" / "20260330"
    build_text = (submit_dir / f"{run_id}_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    predict_text = (submit_dir / f"{run_id}_predict.sbatch").read_text(encoding="utf-8")
    sigma_text = (submit_dir / f"{run_id}_sigma_eval.sbatch").read_text(encoding="utf-8")
    loss_text = (submit_dir / f"{run_id}_training_loss.sbatch").read_text(encoding="utf-8")
    regional_text = (submit_dir / f"{run_id}_regions_step024.sbatch").read_text(encoding="utf-8")
    storm_text = (submit_dir / f"{run_id}_storms_step024.sbatch").read_text(encoding="utf-8")
    spectra_text = (submit_dir / f"{run_id}_spectra.sbatch").read_text(encoding="utf-8")
    finalize_script = submit_dir / f"finalize_lean_eval_{run_id}.sbatch"

    assert 'BUNDLE_PAIRS="20250926:24,20250926:48,20250927:24,20250927:48,20250928:24,20250928:48,20250929:24,20250929:48,20250930:24,20250930:48"' in build_text
    assert f'INPUT_ROOT="{tmp_path / "output" / run_id / "bundles_with_y"}"' in predict_text
    assert 'ALLOW_REBUILT_BUNDLE_ROOT="1"' in predict_text
    assert 'EXPECTED_LANE="o48_o96"' in sigma_text
    assert 'SIGMAS="0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000"' in sigma_text
    assert 'CKPT_NAME="anemoi-by_step-epoch_013-step_100000.ckpt"' in sigma_text
    assert 'NAME_FILTER="forcings_hres"' in loss_text
    assert '"${PYTHON}" "${PROJECT_ROOT}/mlflow/plot.py"' in loss_text
    assert '"${PYTHON}" "${PROJECT_ROOT}/mlflow/plot_loss.py"' in loss_text
    assert 'all_vars.png' in loss_text
    assert 'loss_curves.png' in loss_text
    assert 'REGION_NAMES="amazon_forest_core,eastern_us_coast,andes_central,himalayas_central,maritime_continent,congo_basin"' in regional_text
    assert 'REGION_NAMES="eastern_us_coast,idalia_center"' in storm_text
    assert 'PREDICTIONS_SOURCE_DIR="' in spectra_text
    assert not finalize_script.exists()
    assert "phase=proxy" in result.stdout


def test_o48_o96_helper_continue_full_render_reuses_existing_run_id(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full", run_id_override="manual_existing")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260330"
    build_text = (submit_dir / "manual_existing_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    predict_text = (submit_dir / "manual_existing_predict.sbatch").read_text(encoding="utf-8")

    assert f'RUN_ROOT="{tmp_path / "output" / "manual_existing"}"' in build_text
    assert 'MEMBERS="1"' in build_text
    assert 'BUNDLE_PAIRS=""' in build_text
    assert f'RUN_ID_OVERRIDE="manual_existing"' in predict_text
    assert "phase=continue-full" in result.stdout


def test_o48_o96_helper_proxy_render_normalizes_default_sampler_json(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260330"
    predict_text = (submit_dir / "manual_95a07500_new_o48_o96_20260330_manual_eval_predict.sbatch").read_text(encoding="utf-8")

    rendered = _extract_shell_string(predict_text, "EXTRA_ARGS_JSON")
    parsed = json.loads(rendered)

    assert parsed["schedule_type"] == "experimental_piecewise"
    assert rendered == json.dumps(parsed, separators=(",", ":"))


def test_o48_o96_helper_proxy_render_normalizes_custom_sampler_json(tmp_path: Path):
    custom_sampler = {
        "schedule_type": "experimental_piecewise",
        "num_steps": 28,
        "sigma_max": 1000.0,
        "sigma_transition": 5.0,
        "sigma_min": 0.03,
        "high_schedule_type": "exponential",
        "low_schedule_type": "karras",
        "num_steps_high": 9,
        "num_steps_low": 19,
        "rho": 7.0,
        "sampler": "heun",
        "S_churn": 2.5,
        "S_min": 0.75,
        "S_max": 1000.0,
        "S_noise": 1.05,
    }
    result = _run_helper(
        tmp_path,
        phase="proxy",
        extra_env={"SAMPLER_JSON": json.dumps(custom_sampler, indent=2)},
    )
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260330"
    predict_text = (submit_dir / "manual_95a07500_new_o48_o96_20260330_manual_eval_predict.sbatch").read_text(encoding="utf-8")

    rendered = _extract_shell_string(predict_text, "EXTRA_ARGS_JSON")

    assert rendered == json.dumps(custom_sampler, separators=(",", ":"))
    assert json.loads(rendered) == custom_sampler


def test_o48_o96_helper_missing_source_gribs_fails_fast(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy", populate_source_gribs=False)
    assert result.returncode != 0
    assert "Missing required source GRIB" in result.stderr


def test_finalize_lean_layout_moves_o48_outputs_into_data_without_symlink_clutter(tmp_path: Path):
    run_root = tmp_path / "manual_test_o48"
    run_root.mkdir()
    (run_root / "predictions").mkdir()
    (run_root / "predictions" / "predictions_20250928_step024.nc").write_text("stub\n", encoding="utf-8")
    (run_root / "logs").mkdir()
    (run_root / "bundles_with_y").mkdir()
    (run_root / "EXPERIMENT_CONFIG.yaml").write_text("lane: o48_o96\n", encoding="utf-8")
    (run_root / "o48_o96_metrics.json").write_text("{}\n", encoding="utf-8")
    (run_root / "surface_loss_summary.json").write_text("{}\n", encoding="utf-8")
    for step in ("024", "120"):
        plot_dir = run_root / f"local_plots_regions_step{step}"
        plot_dir.mkdir()
        (plot_dir / "all_regions_plots.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    rendered = FINALIZE_TEMPLATE.read_text(encoding="utf-8")
    rendered = rendered.replace('RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"', f'RUN_ROOT="{run_root}"')
    rendered = rendered.replace('RUN_ID="REPLACE_RUN_ID"', 'RUN_ID="manual_test_o48"')
    rendered = rendered.replace('CREATE_BACKCOMPAT_SYMLINKS="${CREATE_BACKCOMPAT_SYMLINKS:-1}"', 'CREATE_BACKCOMPAT_SYMLINKS="0"')

    finalize_script = tmp_path / "finalize_test.sbatch"
    finalize_script.write_text(rendered, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(finalize_script)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert (run_root / "data" / "predictions").is_dir()
    assert (run_root / "data" / "bundles_with_y").is_dir()
    assert (run_root / "data" / "logs").is_dir()
    assert (run_root / "data" / "o48_o96_metrics.json").is_file()
    assert (run_root / "data" / "surface_loss_summary.json").is_file()
    assert (run_root / "data" / "local_plots_regions_step024").is_dir()
    assert (run_root / "data" / "local_plots_regions_step120").is_dir()
    assert not (run_root / "predictions").exists()
    assert not (run_root / "bundles_with_y").exists()
    assert not (run_root / "logs").exists()
    assert not (run_root / "o48_o96_metrics.json").exists()
    assert not (run_root / "surface_loss_summary.json").exists()
    assert not (run_root / "local_plots_regions_step024").exists()
    assert not (run_root / "local_plots_regions_step120").exists()
    assert (run_root / "local_plots_step024.pdf").is_file()
    assert (run_root / "local_plots_step120.pdf").is_file()
    assert (run_root / "EXPERIMENT_CONFIG.yaml").is_file()
