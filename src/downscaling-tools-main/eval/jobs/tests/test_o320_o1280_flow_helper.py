from __future__ import annotations

import ast
import json
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh"


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
            f"eefo_o320_0001_date{date}_time0000_mem1to10_step24to120_sfc.grib",
            f"eefo_o320_0001_date{date}_time0000_mem1to10_step24to120_pl.grib",
            f"enfo_o1280_0001_date{date}_time0000_step24to120_sfc.grib",
            f"enfo_o1280_0001_date{date}_time0000_mem1to10_step24to120_sfc_y.grib",
            f"enfo_o1280_0001_date{date}_time0000_mem1to10_step24to120_pl_y.grib",
        ):
            (root / name).write_text("stub\n", encoding="utf-8")


def _write_profile_json(path: Path, *, checkpoint_path: Path, host_family: str) -> Path:
    recommended_venv = (
        "/home/ecm5702/dev/.ds-dyn/bin/activate"
        if host_family == "ac"
        else "/home/ecm5702/dev/.ds-ag/bin/activate"
    )
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "stack_flavor": "new",
        "lane": "o320_o1280",
        "host_family": host_family,
        "recommended_venv": recommended_venv,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
        tmp_path / "checkpoints" / "da4d902b71084ecc884a938c4b8930d3" / "anemoi-step.ckpt",
        lres=320,
        hres=1280,
    )
    profile_json = _write_profile_json(
        tmp_path / "profile.json",
        checkpoint_path=fake_checkpoint,
        host_family=host_family,
    )
    _make_fake_hostname(bin_dir / "hostname", host_short)

    if populate_source_gribs:
        if phase == "proxy":
            _touch_required_source_gribs(source_root, ["20230827", "20230828", "20230829", "20230830"])
        else:
            _touch_required_source_gribs(source_root, ["20230826", "20230827", "20230828", "20230829", "20230830"])

    if run_id_override:
        (output_root / run_id_override).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_PATH": str(fake_checkpoint),
            "SOURCE_HPC": "ag",
            "SOURCE_GRIB_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "SUBMIT_ROOT": str(submit_root),
            "PROFILE_JSON_PATH": str(profile_json),
            "NO_SUBMIT": "1",
            "ALLOW_OVERWRITE": "1",
            "RUN_DATE_UTC": "20260327",
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


def test_o320_o1280_helper_proxy_render_uses_truth_bundle_stage(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy")
    assert result.returncode == 0, result.stderr

    run_id = "manual_da4d902b_new_o320_o1280_20260327_manual_eval"
    submit_dir = tmp_path / "submit" / "20260327"
    build_script = submit_dir / f"{run_id}_build_truth_bundles.sbatch"
    predict_script = submit_dir / f"{run_id}_predict.sbatch"

    build_text = build_script.read_text(encoding="utf-8")
    predict_text = predict_script.read_text(encoding="utf-8")
    surface_text = (submit_dir / f"{run_id}_surface_loss.sbatch").read_text(encoding="utf-8")

    assert 'SOURCE_GRIB_ROOT="' in build_text
    assert 'MEMBERS="1"' in build_text
    assert 'BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"' in build_text
    assert f'INPUT_ROOT="{tmp_path / "output" / run_id / "bundles_with_y"}"' in predict_text
    assert 'ALLOW_REBUILT_BUNDLE_ROOT="1"' in predict_text
    assert f'RUN_ID_OVERRIDE="{run_id}"' in predict_text
    assert 'MEMBERS="1"' in predict_text
    assert f'PREDICTIONS_DIR="{tmp_path / "output" / run_id / "predictions"}"' in surface_text
    assert f'OUT_JSON="{tmp_path / "output" / run_id / "data" / "surface_loss.json"}"' in surface_text
    assert "run_surface_loss=1" in result.stdout
    assert f"{run_id}_surface_loss.sbatch" in result.stdout
    assert "phase=proxy" in result.stdout


def test_o320_o1280_helper_continue_full_render_reuses_existing_run_id(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full", run_id_override="manual_existing")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260327"
    build_script = submit_dir / "manual_existing_build_truth_bundles.sbatch"
    predict_script = submit_dir / "manual_existing_predict.sbatch"

    build_text = build_script.read_text(encoding="utf-8")
    predict_text = predict_script.read_text(encoding="utf-8")

    assert f'RUN_ROOT="{tmp_path / "output" / "manual_existing"}"' in build_text
    assert 'MEMBERS="1,2,3,4,5,6,7,8,9,10"' in build_text
    assert 'BUNDLE_PAIRS=""' in build_text
    assert f'RUN_ID_OVERRIDE="manual_existing"' in predict_text
    assert 'MEMBERS="1,2,3,4,5,6,7,8,9,10"' in predict_text
    assert "phase=continue-full" in result.stdout


def test_o320_o1280_helper_ac_proxy_render_uses_ecmwf_and_cpu_safe_jobs(tmp_path: Path):
    if not Path("/home/ecm5702/dev/.ds-dyn/bin/python").exists():
        pytest.skip("ac profile python is unavailable on this host")

    result = _run_helper(
        tmp_path,
        phase="proxy",
        host_family="ac",
        host_short="ac6-test",
    )
    assert result.returncode == 0, result.stderr

    run_id = "manual_da4d902b_new_o320_o1280_20260327_manual_eval"
    submit_dir = tmp_path / "submit" / "20260327"
    build_text = (submit_dir / f"{run_id}_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    local_text = (submit_dir / f"{run_id}_local_plots.sbatch").read_text(encoding="utf-8")
    spectra_text = (submit_dir / f"{run_id}_spectra.sbatch").read_text(encoding="utf-8")
    surface_text = (submit_dir / f"{run_id}_surface_loss.sbatch").read_text(encoding="utf-8")
    tc_text = (submit_dir / f"{run_id}_tc_eval.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --qos=nf" in build_text
    assert "#SBATCH --qos=nf" in local_text
    assert "#SBATCH --qos=nf" in surface_text
    assert "#SBATCH --qos=nf" in tc_text
    assert "#SBATCH --gpus-per-node=" not in build_text
    assert "#SBATCH --gpus-per-node=" not in local_text
    assert "#SBATCH --gpus-per-node=" not in surface_text
    assert "#SBATCH --gpus-per-node=" not in tc_text
    assert 'PREDICTIONS_DIR="' in spectra_text
    assert 'SUBSET_DIR="' not in spectra_text
    assert 'SUPPORT_MODE="regridded"' in tc_text


def test_o320_o1280_helper_can_disable_surface_loss_render(tmp_path: Path):
    result = _run_helper(
        tmp_path,
        phase="proxy",
        extra_env={"RUN_SURFACE_LOSS": "0"},
    )
    assert result.returncode == 0, result.stderr

    run_id = "manual_da4d902b_new_o320_o1280_20260327_manual_eval"
    submit_dir = tmp_path / "submit" / "20260327"

    assert not (submit_dir / f"{run_id}_surface_loss.sbatch").exists()
    assert "run_surface_loss=0" in result.stdout
    assert f"{run_id}_surface_loss.sbatch" not in result.stdout


def test_o320_o1280_helper_proxy_render_normalizes_default_sampler_json(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy")
    assert result.returncode == 0, result.stderr

    submit_dir = tmp_path / "submit" / "20260327"
    predict_text = (submit_dir / f"manual_da4d902b_new_o320_o1280_20260327_manual_eval_predict.sbatch").read_text(encoding="utf-8")

    rendered = _extract_shell_string(predict_text, "EXTRA_ARGS_JSON")
    parsed = json.loads(rendered)

    assert parsed["schedule_type"] == "experimental_piecewise"
    assert rendered == json.dumps(parsed, separators=(",", ":"))


def test_o320_o1280_helper_proxy_render_normalizes_custom_sampler_json(tmp_path: Path):
    custom_sampler = {
        "schedule_type": "experimental_piecewise",
        "num_steps": 23,
        "sigma_max": 1000.0,
        "sigma_transition": 7.0,
        "sigma_min": 0.03,
        "high_schedule_type": "exponential",
        "low_schedule_type": "karras",
        "num_steps_high": 7,
        "num_steps_low": 16,
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

    submit_dir = tmp_path / "submit" / "20260327"
    predict_text = (submit_dir / f"manual_da4d902b_new_o320_o1280_20260327_manual_eval_predict.sbatch").read_text(encoding="utf-8")

    rendered = _extract_shell_string(predict_text, "EXTRA_ARGS_JSON")

    assert rendered == json.dumps(custom_sampler, separators=(",", ":"))
    assert json.loads(rendered) == custom_sampler


def test_o320_o1280_helper_continue_full_requires_run_id_override(tmp_path: Path):
    result = _run_helper(tmp_path, phase="continue-full")
    assert result.returncode != 0
    assert "PHASE=continue-full requires RUN_ID_OVERRIDE" in result.stderr


def test_o320_o1280_helper_missing_source_gribs_fails_fast(tmp_path: Path):
    result = _run_helper(tmp_path, phase="proxy", populate_source_gribs=False)
    assert result.returncode != 0
    assert "Missing required source GRIB" in result.stderr
