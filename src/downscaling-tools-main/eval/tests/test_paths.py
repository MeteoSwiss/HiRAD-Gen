from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _reload_paths():
    sys.modules.pop("eval.paths", None)
    return importlib.import_module("eval.paths")


def test_paths_defaults_to_scratch_eval(monkeypatch):
    monkeypatch.delenv("EVAL_ROOT", raising=False)
    paths = _reload_paths()

    assert paths.DEFAULT_EVAL_ROOT == "/home/ecm5702/scratch/eval"
    assert paths.DEFAULT_SCOREBOARD_DIR == "/home/ecm5702/scratch/eval/scoreboards"
    assert paths.DEFAULT_SLURM_OUTPUT_PATTERN == "/home/ecm5702/scratch/eval/_slurm/%x_%j.out"
    assert paths.LEGACY_EVAL_ROOT == "/home/ecm5702/perm/eval"


def test_paths_honors_eval_root_override(monkeypatch, tmp_path: Path):
    override = tmp_path / "custom_eval"
    monkeypatch.setenv("EVAL_ROOT", str(override))
    paths = _reload_paths()

    assert paths.DEFAULT_EVAL_ROOT == str(override)
    assert paths.resolve_eval_root() == override
    assert paths.default_scoreboard_path("example.md") == str(override / "scoreboards" / "example.md")
