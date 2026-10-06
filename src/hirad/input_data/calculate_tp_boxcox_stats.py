"""Compute Box-Cox(lambda) mean/std of a channel over an anemoi dataset (for the tp
normalization constants used by the dataset configs' transform_input_means/stdevs).

Streams over base_dates (trajectories layout) or dates (gridded layout) in float64, and
supports sampling a periodic subset (``--stride``) so the estimate spans all seasons rather
than a short contiguous window. Matches transform_channel in calculate_transformed_stats.py:
    x = clip(x, 0, None); (x**lambda - 1) / lambda

Examples
--------
# IFS-HRES input tp over ~every 8th init of the full trajectories zarr:
python -m hirad.input_data.calculate_tp_boxcox_stats \
    --zarr /capstor/scratch/cscs/pstamenk/ifs-hres-realch1/ifs_hres_traj_2020_202502.zarr \
    --channel tp --stride 8
"""
from __future__ import annotations

import argparse
import numpy as np
from anemoi.datasets import open_dataset


def box_cox(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return (np.power(x, lmbda) - 1.0) / lmbda


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True, help="anemoi dataset path")
    p.add_argument("--channel", default="tp", help="variable name to transform (default tp)")
    p.add_argument("--lmbda", type=float, default=0.25, help="Box-Cox lambda (default 0.25)")
    p.add_argument("--stride", type=int, default=1,
                   help="use every STRIDE-th base_date/date (default 1 = all). Pick STRIDE coprime "
                        "with the per-day step count so samples sweep all hours of day (for hourly "
                        "data avoid multiples of 24 -> a prime like 97 is safest; for a 2/day base "
                        "axis use an odd stride like 7). Otherwise the sample aliases onto one hour.")
    p.add_argument("--start", default=None, help="restrict to dates >= START (gridded datasets)")
    p.add_argument("--end", default=None, help="restrict to dates <= END (gridded datasets)")
    p.add_argument("--select", action="store_true",
                   help="open only --channel (efficient for wide gridded targets; do NOT use on "
                        "the 5D forecast/trajectories store)")
    args = p.parse_args()

    open_kwargs = {}
    if args.start is not None:
        open_kwargs["start"] = args.start
    if args.end is not None:
        open_kwargs["end"] = args.end
    if args.select:
        open_kwargs["select"] = [args.channel]
    ds = open_dataset(args.zarr, **open_kwargs)
    ch_idx = list(ds.variables).index(args.channel)
    is_traj = hasattr(ds, "base_dates") and ds.data.ndim == 5 if hasattr(ds, "data") else len(ds.shape) == 5
    n_outer = ds.shape[0]
    sel = list(range(0, n_outer, args.stride))
    dates = np.asarray(ds.base_dates if is_traj else ds.dates)
    print(f"{args.zarr}\n shape={ds.shape} channel={args.channel}(idx {ch_idx}) lambda={args.lmbda}")
    print(f" sampling {len(sel)}/{n_outer} {'base_dates' if is_traj else 'dates'} "
          f"({dates[sel[0]]} .. {dates[sel[-1]]}, stride {args.stride})")

    n = 0
    s = 0.0
    ss = 0.0
    for i in sel:
        # trajectories: (var, ens, step, cell); gridded: (var, ens, cell)
        arr = ds[i]
        x = box_cox(arr[ch_idx], args.lmbda)
        n += x.size
        s += x.sum()
        ss += (x * x).sum()

    mean = s / n
    std = float(np.sqrt(ss / n - mean * mean))
    key = f"{args.channel}-box_cox_{str(args.lmbda).replace('0.', '0')}"
    print(f"\n{args.channel} Box-Cox({args.lmbda}): mean={mean:.10f} std={std:.10f} (N={n})")
    print("\n--- paste into the dataset config ---")
    print(f"transform_input_means: {{'{key}': {mean}}}")
    print(f"transform_input_stdevs: {{'{key}': {std}}}")


if __name__ == "__main__":
    main()
