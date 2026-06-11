"""Scan an anemoi dataset for timesteps containing NaN values.

Usage:
    python src/anemoi-core-ds-collab/datasets/scan_nan_timestamps.py \
        --dataset /capstor/store/cscs/pasc/c38/anemoi-downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr \
        --start_date 2019-01-01 \
        --end_date 2019-12-31 \
        --variables 2t 10u 10v tp tcw u_500 u_850 v_500 v_850 t_500 t_850 z_500 z_850
"""

import argparse
import logging
import numpy as np
from anemoi.datasets import open_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)


def scan_nans(dataset_path: str, start_date: np.datetime64, end_date: np.datetime64, variables: list[str]):
    ds = open_dataset(dataset_path)

    # Resolve variable indices (skip any not present in dataset)
    var_indices = {}
    for v in variables:
        if v in ds.variables:
            var_indices[v] = ds.variables.index(v)
        else:
            LOGGER.warning("Variable %s not found in dataset, skipping.", v)

    if not var_indices:
        raise ValueError("None of the requested variables are in the dataset.")

    LOGGER.info("Dataset variables: %s", ds.variables)
    LOGGER.info("Scanning variables: %s", list(var_indices.keys()))
    LOGGER.info("Dataset date range: %s — %s", ds.dates[0], ds.dates[-1])

    # Find index range for the requested window
    start_i = None
    end_i = None
    for i, d in enumerate(ds.dates):
        if start_i is None and d >= start_date:
            start_i = i
        if d <= end_date:
            end_i = i

    if start_i is None or end_i is None or start_i > end_i:
        raise ValueError(f"No dates found between {start_date} and {end_date}")

    LOGGER.info(
        "Scanning %d timesteps (indices %d–%d): %s — %s",
        end_i - start_i + 1,
        start_i,
        end_i,
        ds.dates[start_i],
        ds.dates[end_i],
    )

    nan_timestamps = []
    idx_list = list(var_indices.values())

    for i in range(start_i, end_i + 1):
        # ds[i] shape: (variables, ensemble, gridpoints) or (variables, gridpoints)
        sample = ds[i]  # numpy array
        # Select only the variables we care about
        chunk = sample[idx_list]  # (n_vars, ...)
        if np.isnan(chunk).any():
            nan_count = int(np.isnan(chunk).sum())
            nan_vars = [v for v, idx in var_indices.items() if np.isnan(sample[idx]).any()]
            LOGGER.warning("NaN found at %s  count=%d  affected_vars=%s", ds.dates[i], nan_count, nan_vars)
            nan_timestamps.append(str(ds.dates[i]))

        if (i - start_i) % 500 == 0:
            LOGGER.info("Progress: %d / %d", i - start_i, end_i - start_i)

    LOGGER.info("Scan complete. %d NaN timestep(s) found.", len(nan_timestamps))
    if nan_timestamps:
        print("\nNaN timestamps (paste into missing_dates):")
        for ts in nan_timestamps:
            print(f"  - {ts}")
    else:
        print("\nNo NaN values found in the requested window.")

    return nan_timestamps


def main():
    parser = argparse.ArgumentParser(description="Scan an anemoi dataset for NaN timesteps.")
    parser.add_argument("--dataset", required=True, help="Path to the zarr dataset")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["2t", "10u", "10v", "tp", "tcw", "u_500", "u_850", "v_500", "v_850", "t_500", "t_850", "z_500", "z_850"],
        help="Variables to check (default: all ERA-COSMO lres variables)",
    )
    args = parser.parse_args()

    scan_nans(
        dataset_path=args.dataset,
        start_date=np.datetime64(args.start_date),
        end_date=np.datetime64(args.end_date),
        variables=args.variables,
    )


if __name__ == "__main__":
    main()
