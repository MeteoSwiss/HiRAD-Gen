from __future__ import annotations

import os
from pathlib import Path

EVAL_ROOT_ENV = "EVAL_ROOT"
CANONICAL_EVAL_ROOT = "/home/ecm5702/scratch/eval"
LEGACY_EVAL_ROOT = "/home/ecm5702/perm/eval"

REFERENCE_ROOT_ENV = "REFERENCE_ROOT"
CANONICAL_REFERENCE_ROOT = "/home/ecm5702/perm/reference"
DEFAULT_REFERENCE_LANE = "o96_o320"

SPECTRA_MODEL_LANES = {
    "eefo_o96": "o48_o96",
    "enfo_o320": "o96_o320",
    "eefo_o320": "o96_o320",
    "enfo_o1280": "o320_o1280",
    "destine_o2560_i4ql": "o1280_o2560",
    "destine_o2560_i4ql_step72_100d": "o1280_o2560",
    "destine_o2560_i4ql_step72_subset10rnd_20260303": "o1280_o2560",
}


def default_eval_root() -> str:
    return os.environ.get(EVAL_ROOT_ENV, CANONICAL_EVAL_ROOT)


def resolve_eval_root(path: str | Path | None = None) -> Path:
    raw = path if path is not None else default_eval_root()
    return Path(raw).expanduser()


def default_scoreboard_dir() -> str:
    return str(resolve_eval_root() / "scoreboards")


def default_scoreboard_path(name: str) -> str:
    return str(resolve_eval_root() / "scoreboards" / name)


def default_slurm_dir() -> str:
    return str(resolve_eval_root() / "_slurm")


def default_slurm_output_pattern() -> str:
    return str(resolve_eval_root() / "_slurm" / "%x_%j.out")


def default_reference_root() -> str:
    return os.environ.get(REFERENCE_ROOT_ENV, CANONICAL_REFERENCE_ROOT)


def resolve_reference_root(path: str | Path | None = None) -> Path:
    raw = path if path is not None else default_reference_root()
    return Path(raw).expanduser()


def reference_lane_dir(lane: str = DEFAULT_REFERENCE_LANE, *, kind: str | None = None, root: str | Path | None = None) -> Path:
    path = resolve_reference_root(root) / lane
    return path / kind if kind else path


def reference_spectra_base(*, root: str | Path | None = None) -> Path:
    return resolve_reference_root(root) / "spectra"


def reference_spectra_dir(name: str, *, lane: str | None = None, root: str | Path | None = None) -> Path:
    raw = Path(name).expanduser()
    if raw.is_absolute() or len(raw.parts) > 1:
        return raw
    resolved_lane = lane or SPECTRA_MODEL_LANES.get(name)
    if resolved_lane:
        return reference_lane_dir(resolved_lane, kind="spectra", root=root) / name
    return reference_spectra_base(root=root) / name


def reference_tc_dir(lane: str = DEFAULT_REFERENCE_LANE, *, root: str | Path | None = None) -> Path:
    return reference_lane_dir(lane, kind="tc", root=root)


def reference_tc_event_dir(event: str, lane: str = DEFAULT_REFERENCE_LANE, *, root: str | Path | None = None) -> Path:
    return reference_tc_dir(lane, root=root) / event


def reference_grib_dir(lane: str = DEFAULT_REFERENCE_LANE, *, root: str | Path | None = None) -> Path:
    return reference_lane_dir(lane, kind="grib", root=root)


DEFAULT_EVAL_ROOT = default_eval_root()
DEFAULT_SCOREBOARD_DIR = default_scoreboard_dir()
DEFAULT_SLURM_DIR = default_slurm_dir()
DEFAULT_SLURM_OUTPUT_PATTERN = default_slurm_output_pattern()

DEFAULT_REFERENCE_ROOT = default_reference_root()
DEFAULT_REFERENCE_SPECTRA_BASE = str(reference_spectra_base())
DEFAULT_REFERENCE_TC_DIR = str(reference_tc_dir())
