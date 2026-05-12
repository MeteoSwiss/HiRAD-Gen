from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "eval/jobs/templates/submit_o1280_o2560_manual_eval_flow.sh"


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
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


def _touch_required_source_gribs(input_root: Path, forcing_root: Path, dates: list[str]) -> None:
    for date in dates:
        (input_root / f"enfo_o1280_0001_date{date}_time0000_step006to120by006_input.grib").write_text("stub\n", encoding="utf-8")
        (forcing_root / f"destine_rd_fc_oper_i4ql_o2560_date{date}_time0000_step006to120by006_sfc.grib").write_text("stub\n", encoding="utf-8")
        (forcing_root / f"destine_rd_fc_oper_i4ql_o2560_date{date}_time0000_step006to120by006_y.grib").write_text("stub\n", encoding="utf-8")


def _run_helper(
    tmp_path: Path,
    *,
    strict_bundle_ready: bool,
    allow_debug_fallback: bool = False,
    host_short: str = "ag6-test",
) -> subprocess.CompletedProcess[str]:
    input_root = tmp_path / "input"
    forcing_root = tmp_path / "forcing"
    submit_root = tmp_path / "submit"
    output_root = tmp_path / "output"
    bin_dir = tmp_path / "bin"
    ckpt_dir = tmp_path / "checkpoints" / "811960e67c43455a9eee810ade1ad10f"

    input_root.mkdir(parents=True, exist_ok=True)
    forcing_root.mkdir(parents=True, exist_ok=True)
    submit_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _touch_required_source_gribs(
        input_root,
        forcing_root,
        ["20241108", "20241109", "20241110", "20241111", "20241112"],
    )
    _make_fake_hostname(bin_dir / "hostname", host_short)

    base_ckpt = ckpt_dir / "anemoi-by_epoch-epoch_028-step_033375.ckpt"
    inference_ckpt = ckpt_dir / "inference-anemoi-by_epoch-epoch_028-step_033375.ckpt"
    base_ckpt.write_text("stub\n", encoding="utf-8")
    inference_ckpt.write_text("stub\n", encoding="utf-8")

    profile_json = tmp_path / "profile.json"
    profile_json.write_text(
        json.dumps(
            {
                "checkpoint_path": str(inference_ckpt),
                "stack_flavor": "new",
                "lane": "o1280_o2560",
                "host_family": "ag",
                "recommended_venv": "/home/ecm5702/dev/.ds-ag/bin/activate",
            }
        ),
        encoding="utf-8",
    )
    preflight_json = tmp_path / "preflight.json"
    preflight_json.write_text(
        json.dumps(
            {
                "strict_bundle_ready": strict_bundle_ready,
                "proof_only_ready": True,
                "blocker_summary": "none" if strict_bundle_ready else "Missing low-res surface variables: msl",
                "contract": {
                    "output_weather_state_mode": "all",
                    "output_weather_states_csv": "10u,10v,2t,msl",
                    "plot_weather_states_csv": "10u,10v,2t,msl",
                    "spectra_weather_states_csv": "10u,10v,2t,msl",
                    "num_gpus_per_model": 4,
                    "slim_output": True,
                },
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_PATH": str(base_ckpt),
            "SOURCE_HPC": "ag",
            "SOURCE_INPUT_ROOT": str(input_root),
            "SOURCE_FORCING_ROOT": str(forcing_root),
            "OUTPUT_ROOT": str(output_root),
            "SUBMIT_ROOT": str(submit_root),
            "RUN_DATE_UTC": "20260421",
            "RUN_SUFFIX": "manual_eval",
            "PROOF_BUNDLE_PAIRS": "20241108:120",
            "PROFILE_JSON_PATH": str(profile_json),
            "PREFLIGHT_JSON_PATH": str(preflight_json),
            "NO_SUBMIT": "1",
            "ALLOW_OVERWRITE": "1",
            "ALLOW_DEBUG_FALLBACK": "1" if allow_debug_fallback else "0",
        }
    )
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    return subprocess.run(
        ["bash", str(HELPER)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_o1280_o2560_helper_renders_strict_bundle_flow(tmp_path: Path):
    result = _run_helper(tmp_path, strict_bundle_ready=True)
    assert result.returncode == 0, result.stderr

    run_id = "manual_811960e6_new_o1280_o2560_20260421_manual_eval"
    submit_dir = tmp_path / "submit" / "20260421"
    build_text = (submit_dir / f"{run_id}_build_truth_bundles.sbatch").read_text(encoding="utf-8")
    predict_text = (submit_dir / f"{run_id}_predict.sbatch").read_text(encoding="utf-8")
    local_text = (submit_dir / f"{run_id}_local_plots.sbatch").read_text(encoding="utf-8")
    spectra_text = (submit_dir / f"{run_id}_spectra.sbatch").read_text(encoding="utf-8")
    surface_text = (submit_dir / f"{run_id}_surface_loss.sbatch").read_text(encoding="utf-8")

    assert 'SOURCE_INPUT_ROOT="' in build_text
    assert 'SOURCE_FORCING_ROOT="' in build_text
    assert 'BUNDLE_PAIRS="20241108:120"' in build_text
    assert f'INPUT_ROOT="{tmp_path / "output" / run_id / "bundles_with_y"}"' in predict_text
    assert 'NUM_GPUS_PER_MODEL="4"' in predict_text
    assert 'OUTPUT_WEATHER_STATES="10u,10v,2t,msl"' in predict_text
    assert 'SLIM_OUTPUT="1"' in predict_text
    assert 'WEATHER_STATES="10u,10v,2t,msl"' in local_text
    assert 'WEATHER_STATES="10u,10v,2t,msl"' in spectra_text
    assert f'PREDICTIONS_DIR="{tmp_path / "output" / run_id / "predictions"}"' in surface_text
    assert f'OUT_JSON="{tmp_path / "output" / run_id / "surface_loss_summary.json"}"' in surface_text
    assert "strict_bundle_ready=1" in result.stdout


def test_o1280_o2560_helper_renders_debug_fallback_when_allowed(tmp_path: Path):
    result = _run_helper(tmp_path, strict_bundle_ready=False, allow_debug_fallback=True)
    assert result.returncode == 0, result.stderr

    run_id = "manual_811960e6_new_o1280_o2560_20260421_manual_eval"
    submit_dir = tmp_path / "submit" / "20260421"
    debug_text = (submit_dir / f"{run_id}_debug_dataloader.sbatch").read_text(encoding="utf-8")

    assert 'NTASKS="4"' in debug_text
    assert 'EXTRA_PREDICT_FLAGS="--output-weather-state-mode all --output-weather-states 10u,10v,2t,msl --slim-output"' in debug_text
    assert 'PLOT_WEATHER_STATES="10u,10v,2t,msl"' in debug_text
    assert "strict_bundle_ready=0" in result.stdout
