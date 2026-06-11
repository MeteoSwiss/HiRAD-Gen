from __future__ import annotations

import argparse
import json
from pathlib import Path

import xarray as xr

from eval.archive import run as eval_run


def test_default_checkpoint_run_name():
    name = eval_run._default_checkpoint_run_name("my-exp", "/root/ckpt")
    assert name == "my-exp-last"

    name2 = eval_run._default_checkpoint_run_name(
        "/tmp/checkpoints/exp42/final.ckpt", "/ignored"
    )
    assert name2 == "exp42-final"


def test_run_from_predictions_writes_metadata(tmp_path: Path):
    src = tmp_path / "input_predictions.nc"
    xr.Dataset({"a": ("x", [1, 2, 3])}).to_netcdf(src)

    args = argparse.Namespace(
        predictions_nc=str(src),
        run_name="my_eval",
        eval_root=str(tmp_path / "eval"),
        run_region=False,
    )
    run_dir = eval_run.run_from_predictions(args)
    assert run_dir == tmp_path / "eval" / "my_eval"
    assert (run_dir / "predictions.nc").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "predictions"
    assert metadata["run_name"] == "my_eval"
    assert metadata["predictions_nc"] == str(run_dir / "predictions.nc")
