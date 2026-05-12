"""Tests for eval.jobs.renderer."""

from __future__ import annotations

import subprocess

from eval.jobs.renderer import render_sbatch


def test_render_sbatch_default_resources(tmp_path):
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
    )
    assert "#SBATCH --qos=nf" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --cpus-per-task=16" in script
    # No GPU directive since qos=nf and gpus=0
    assert "--gpus-per-node" not in script

    path = tmp_path / "test.sbatch"
    path.write_text(script)
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_render_sbatch_with_resource_overrides():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"gpus": 2, "time": "08:00:00", "mem": "256G"},
    )
    assert "#SBATCH --time=08:00:00" in script
    assert "#SBATCH --mem=256G" in script
    assert "#SBATCH --gpus-per-node=2" in script


def test_render_sbatch_gpu_directive():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"gpus": 1},
    )
    assert "#SBATCH --gpus-per-node=1" in script


def test_render_sbatch_ng_no_gpu_directive():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"qos": "ng", "gpus": 0},
    )
    assert "#SBATCH --qos=ng" in script
    assert "#SBATCH --gpus-per-node=0" in script


def test_render_sbatch_job_name_suffix():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        resource_overrides={"job_name_suffix": "-predict"},
    )
    assert "-predict-" in script
    assert "#SBATCH --job-name=eval-o96_o320-predict-" in script


def test_render_sbatch_cli_overrides():
    script = render_sbatch(
        lane="o96_o320", host="atos_ac", checkpoint="/tmp/test.ckpt",
        overrides={"--only": "tc"},
    )
    assert "--only" in script
    assert "tc" in script
