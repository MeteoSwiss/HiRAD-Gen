"""Tests for eval.jobs.pipeline."""

from __future__ import annotations

import subprocess

from eval.jobs.pipeline import render_pipeline


LANE = "o96_o320"
HOST = "atos_ac"
CKPT = "/tmp/test.ckpt"


def test_render_pipeline_script_count():
    manifest = render_pipeline(lane=LANE, host=HOST, checkpoint=CKPT)
    # predict + 4 default evaluators (tc, spectra, surface, region_plot) + scoreboard
    assert len(manifest.scripts) == 6


def test_render_pipeline_dependency_chain():
    manifest = render_pipeline(lane=LANE, host=HOST, checkpoint=CKPT)
    predict = manifest.scripts[0]
    evals = manifest.scripts[1:-1]
    scoreboard = manifest.scripts[-1]

    assert predict.dependencies == []
    for entry in evals:
        assert entry.dependencies == [predict.name]
    assert set(scoreboard.dependencies) == {e.name for e in evals}


def test_render_pipeline_no_split():
    manifest = render_pipeline(
        lane=LANE, host=HOST, checkpoint=CKPT, split_evaluators=False,
    )
    # predict + 1 combined evaluate + scoreboard
    assert len(manifest.scripts) == 3
    names = [s.name for s in manifest.scripts]
    assert "02_evaluate" in names


def test_render_pipeline_writes_files(tmp_path):
    out = tmp_path / "pipeline"
    manifest = render_pipeline(
        lane=LANE, host=HOST, checkpoint=CKPT, output_dir=out,
    )
    for entry in manifest.scripts:
        assert (out / entry.path).exists()
    assert (out / "submit_pipeline.sh").exists()


def test_render_pipeline_submit_script_syntax(tmp_path):
    out = tmp_path / "pipeline"
    render_pipeline(lane=LANE, host=HOST, checkpoint=CKPT, output_dir=out)
    submit = out / "submit_pipeline.sh"
    result = subprocess.run(
        ["bash", "-n", str(submit)], capture_output=True, text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_render_pipeline_preflight_injected():
    manifest = render_pipeline(lane=LANE, host=HOST, checkpoint=CKPT)
    for entry in manifest.scripts:
        assert "preflight_cluster" in entry.content, (
            f"preflight_cluster missing in {entry.name}"
        )
