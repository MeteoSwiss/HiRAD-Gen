#!/usr/bin/env python3
"""Convert a torch-format inference output directory to GRIB, for a path you specify.

Loads previously-saved torch results (written by save_results_as_torch /
save_results with output_format='torch') and writes them out as GRIB via
convert_torch_to_grib. Handles both directory layouts _result_dirs produces:

    flat (no base_time):      TORCH_OUT_DIR/{time_step}/{time_step}-*
    forecast (base_time):     TORCH_OUT_DIR/{base_time}/{time_step}/{time_step}-*

time_step and --base-time each accept zero or more explicit values, or 'all'
(the default) to auto-discover every one present under TORCH_OUT_DIR.

Usage:
    python -m hirad.utils.torch_to_grib TORCH_OUT_DIR [time_step ...]

    TORCH_OUT_DIR   top-level output_path that was passed to save_results
                    when inference ran (not the per-step torch_savedir itself)
    time_step       one or more lead times to convert, or 'all' (default)

    Optional overrides:
        --base-time TIME [TIME ...]
                             one or more forecast init times, or 'all'
                             (default: 'all' if TORCH_OUT_DIR uses the
                             base_time layout, otherwise unused)
        --dataset-cfg PATH   dataset config used for channel metadata
                              (default: src/hirad/conf/dataset/anemoi_era_real_inference.yaml)
        --templates PATH     GRIB template directory
                              (default: ~/evalml/resources/inference/templates)
"""

import argparse
import os
import re

from omegaconf import OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.inference_utils import convert_torch_to_grib

DEFAULT_DATASET_CFG = 'src/hirad/conf/dataset/anemoi_era_real_inference.yaml'
DEFAULT_TEMPLATES = os.path.expanduser('~/evalml/resources/inference/templates')

TIMESTAMP_RE = re.compile(r'^\d{8}-\d{4}$')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('torch_out_dir', help="Torch output directory to convert (the output_path passed to save_results)")
    parser.add_argument('time_step', nargs='*', default=['all'], help="Lead time(s) to convert, or 'all' (default)")
    parser.add_argument('--base-time', nargs='*', default=['all'], help="Forecast init time(s) to convert, or 'all' (default); ignored if TORCH_OUT_DIR isn't base_time-nested")
    parser.add_argument('--dataset-cfg', default=DEFAULT_DATASET_CFG, help="Dataset config used for channel metadata")
    parser.add_argument('--templates', default=DEFAULT_TEMPLATES, help="GRIB template directory")
    return parser.parse_args()


def _timestamp_subdirs(parent):
    """Immediate subdirectories of parent named like a time_step/base_time (YYYYMMDD-HHMM)."""
    return sorted(
        name for name in os.listdir(parent)
        if TIMESTAMP_RE.match(name) and os.path.isdir(os.path.join(parent, name))
    )


def _is_leaf_savedir(dir_path):
    """True if dir_path is a torch_savedir itself, i.e. it directly holds the
    '{name}-target' / '{name}-predictions' / etc. files save_results_as_torch writes,
    rather than further timestamp subdirectories.
    """
    name = os.path.basename(dir_path)
    return any(f.startswith(f'{name}-') and os.path.isfile(os.path.join(dir_path, f)) for f in os.listdir(dir_path))


def resolve_runs(torch_out_dir, time_step_args, base_time_args):
    """Work out the (base_time, time_step) pairs to convert, expanding 'all' /
    omitted args by discovering what's actually on disk under torch_out_dir.
    Returns a list of (base_time_or_None, time_step) tuples.
    """
    top_level = _timestamp_subdirs(torch_out_dir)
    uses_base_time = bool(top_level) and not any(_is_leaf_savedir(os.path.join(torch_out_dir, d)) for d in top_level)

    if not uses_base_time:
        if base_time_args not in (['all'], []):
            raise SystemExit(f"--base-time given but {torch_out_dir} has no base_time-nested layout (no forecast subdirectories found)")
        time_steps = top_level if time_step_args == ['all'] else time_step_args
        return [(None, ts) for ts in time_steps]

    base_times = top_level if base_time_args == ['all'] else base_time_args
    runs = []
    for bt in base_times:
        bt_dir = os.path.join(torch_out_dir, bt)
        if not os.path.isdir(bt_dir):
            raise SystemExit(f"base_time directory not found: {bt_dir}")
        available = _timestamp_subdirs(bt_dir)
        time_steps = available if time_step_args == ['all'] else time_step_args
        runs.extend((bt, ts) for ts in time_steps)
    return runs


def main():
    args = parse_args()

    if not os.path.isdir(args.torch_out_dir):
        raise SystemExit(f"Not a directory: {args.torch_out_dir}")

    runs = resolve_runs(args.torch_out_dir, args.time_step, args.base_time)
    if not runs:
        raise SystemExit(f"No time steps found to convert under {args.torch_out_dir}")

    # DistributedManager.initialize_slurm() reads MASTER_ADDR/PORT from the
    # environment itself rather than the "localhost"/"12355" defaults
    # initialize() computes, so on an interactive SLURM shell (SLURM_PROCID
    # set, no launcher-provided MASTER_ADDR) it crashes unless we set these
    # ourselves. Harmless single-process defaults for this standalone script.
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    DistributedManager.initialize()

    dataset_cfg = OmegaConf.to_container(OmegaConf.load(args.dataset_cfg))
    # times=[]: we validate each run's time_step against dataset.time() ourselves
    # below (so one bad/missing time_step skips just that run, not the whole batch)
    # rather than via get_dataset_and_sampler_inference's all-or-nothing check.
    dataset, _ = get_dataset_and_sampler_inference(dataset_cfg=dataset_cfg, times=[])
    valid_times = set(dataset.time())

    print(f'Converting {len(runs)} run(s) from {args.torch_out_dir} ...\n')
    grib_savedirs = []
    failures = []
    for base_time, time_step in runs:
        label = f'{time_step}' if base_time is None else f'{base_time}/{time_step}'
        if time_step not in valid_times:
            print(f'  [skip] {label}: time_step not found in dataset')
            failures.append(label)
            continue
        try:
            grib_savedir = convert_torch_to_grib(args.torch_out_dir, time_step, dataset, args.templates, base_time=base_time)
            print(f'  [ok]   {label} -> {grib_savedir}')
            grib_savedirs.append(grib_savedir)
        except Exception as e:
            print(f'  [fail] {label}: {e}')
            failures.append(label)

    if failures:
        print(f'\n{len(failures)} of {len(runs)} run(s) failed or were skipped: {", ".join(failures)}')

    for grib_savedir in sorted(set(grib_savedirs)):
        print(f'\nOutput files in {grib_savedir}:')
        for f in sorted(os.listdir(grib_savedir)):
            path = os.path.join(grib_savedir, f)
            print(f'  {f}  ({os.path.getsize(path):,} bytes)')

    # Quick sanity check with earthkit
    try:
        import earthkit.data as ekd
        for grib_savedir in sorted(set(grib_savedirs)):
            for f in sorted(os.listdir(grib_savedir)):
                ds = ekd.from_source('file', os.path.join(grib_savedir, f))
                print(f'\n{f}:')
                print(ds.ls().to_string(index=False))
    except Exception as e:
        print(f'\n(earthkit check skipped: {e})')


if __name__ == '__main__':
    main()
