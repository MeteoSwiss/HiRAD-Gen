"""Build the IFS-HRES trajectories (5D forecast) anemoi zarr from precomputed GRIBs.

Registers the custom `ifs-hres-files` source (by importing anemoi_sources) and
runs the anemoi create tasks in-process (serial), mirroring the CLI
`anemoi-datasets create`. Run inside the container via SLURM.
"""
from __future__ import annotations

import argparse
import inspect
import logging

import numpy as np

import hirad.input_data.anemoi_sources  # noqa: F401  (registers ifs-hres-files source)
import anemoi.datasets.create.recipe.statistics as _stats
from anemoi.datasets.create.tasks import run_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ifs_hres_anemoi")

CORE_TASKS = ["init", "load", "finalise"]


def _patch_trajectory_statistics_filter() -> None:
    """Work around an anemoi-datasets 0.5.41 bug on this numpy.

    ``Statistics.trajectory_statistics_filter`` does ``np.all(np.diff(steps) >= 0)``
    where ``steps`` is a ``timedelta64`` array; comparing timedelta64 to the int
    ``0`` raises a casting error on the container's numpy. Rewrite that one check
    as a timedelta-vs-timedelta comparison; the rest is an exact copy.
    """
    def trajectory_statistics_filter(self, base_dates, steps):
        base_dates = np.asarray(base_dates)
        steps = np.asarray(steps)
        assert np.all(steps[1:] >= steps[:-1]), f"steps must be sorted ascending, got {steps}"
        valid_times = np.unique((base_dates[:, None] + steps[None, :]).ravel())
        start, end = self.statistics_dates(list(valid_times))
        np_start = np.datetime64(start).astype(base_dates.dtype)
        np_end = np.datetime64(end).astype(base_dates.dtype)
        _stats.LOG.info(f"Using trajectory statistics envelope: start={np_start}, end={np_end}")
        return _stats.TrajectoryStatisticsFilter(np_start, np_end, steps[0], steps[-1])

    _stats.Statistics.trajectory_statistics_filter = trajectory_statistics_filter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", required=True)
    p.add_argument("--path", required=True, help="output .zarr path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tasks", default=",".join(CORE_TASKS), help="comma-separated create tasks")
    args = p.parse_args()

    logger.info("run_task signature: %s", inspect.signature(run_task))
    _patch_trajectory_statistics_filter()
    opts = dict(recipe=args.recipe, path=args.path, overwrite=args.overwrite)
    for t in args.tasks.split(","):
        logger.info("=== task %s ===", t)
        run_task(t, **opts)
    logger.info("build done -> %s", args.path)


if __name__ == "__main__":
    main()
