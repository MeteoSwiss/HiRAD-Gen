from anemoi.datasets import open_dataset
from scipy.sparse import csr_matrix
import numpy as np
import argparse
import logging

# python src/anemoi-core-ds-collab/datasets/compute_residuals.py --lres_dataset=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr --hres_dataset=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/mch-co2-an-archive-0p02-2015-2020-1h-v3-pl13.zarr --interpolation_matrix=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/era2cosmo_cropped_to_cosmo_linear.mat.npz --output_path=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/tmp2 --start_date=2016-01-01 --end_date=2016-01-05

def compute_residuals(lres_dataset_path: str, hres_dataset_path: str, interpolation_matrix_path: str, output_path: str, start_date: np.datetime64, end_date: np.datetime64):
    logging.info('opening datasets')
    lres_ds = open_dataset(lres_dataset_path)
    hres_ds = open_dataset(hres_dataset_path)
    mat = np.load(interpolation_matrix_path)
    M = csr_matrix((mat['data'], mat['indices'], mat['indptr']), shape=mat['shape'])

    logging.info('metadata')
    variables = list(set(lres_ds.variables).intersection(set(hres_ds.variables)))
    variables.sort()
    hres_var_indices = []
    lres_var_indices = []
    for v in variables:
        hres_var_indices.append(hres_ds.variables.index(v))
        lres_var_indices.append(lres_ds.variables.index(v))

    lres_start_index = -1
    lres_end_index = -1
    hres_start_index = -1
    hres_end_index = -1
    for i in range(len(hres_ds.dates)):
        d = hres_ds.dates[i]
        if d == start_date:
            hres_start_index = i
        if d > end_date and hres_end_index == -1:
            hres_end_index = i

    for i in range(len(lres_ds.dates)):
        d = lres_ds.dates[i]
        if d == start_date:
            lres_start_index = i
        if d > end_date and lres_end_index == -1:
            lres_end_index = i

    means = np.zeros(len(variables))
    sums = np.zeros(len(variables))
    # residuals should be small enough that we don't need to initialize with MAXINT
    maxes = np.ones(len(variables)) * - 999999
    mins = np.ones(len(variables)) * 999999
    stdevs = np.zeros(len(variables))

    # iterate through dates in dataset to get maxes, mins, and means
    h_index = hres_start_index
    l_index = lres_start_index
    logging.info('computing means,mins,maxes')
    while h_index < hres_end_index and l_index < lres_end_index:
        logging.info(hres_ds.dates[h_index])
        hres_values = hres_ds.data[h_index, hres_var_indices, 0, :]
        lres_values = lres_ds.data[l_index, lres_var_indices, 0, :]
        # get interpolated values
        lres_interp_values = interpolate(lres_values, M)

        # compute residuals
        residuals = hres_values - lres_interp_values
        # get minimum/max/sum for each variable
        local_min = np.min(residuals, 1)
        local_max = np.max(residuals, 1)
        local_sum = np.sum(residuals, 1)
        sums = sums + local_sum
        maxes = np.maximum(maxes, local_max)
        mins = np.minimum(mins, local_min)

        h_index = h_index + 1
        l_index = l_index + 1
    
    # calculate means
    denom = (hres_end_index - hres_start_index) * hres_ds.shape[-1]
    means = sums / denom
    
    sum_residuals_diff = np.zeros(len(variables))

    logging.info('computing stdev')

    # using means, calculate standard devs
    h_index = hres_start_index
    l_index = lres_start_index
    while h_index < hres_end_index and l_index < lres_end_index:
        
        logging.info(hres_ds.dates[h_index])

        hres_values = hres_ds.data[h_index, hres_var_indices, 0, :]
        lres_values = lres_ds.data[l_index, lres_var_indices, 0, :]
        # get interpolated values
        lres_interp_values = interpolate(lres_values, M)

        # compute residuals
        residuals = hres_values - lres_interp_values

        # get diff for each variable. maybe there's a np method for this but I don't know it.
        for i in range(lres_interp_values.shape[0]):
            sum_residuals_diff[i] = sum_residuals_diff[i] + np.sum(np.pow(residuals[i,:] - means[i], 2))

        h_index = h_index + 1
        l_index = l_index + 1
    
    sigma_2 = sum_residuals_diff / denom
    stdevs = np.pow(sigma_2, 0.5)

    save_stats(output_path, variables, means=means, maxes=maxes, mins=mins, stdevs=stdevs)


def interpolate(lres_vals: np.ndarray, M):
    interp_lres_vals = np.zeros([lres_vals.shape[0], M.shape[0]])
    for i in range(lres_vals.shape[0]):
        interp_lres_vals[i,:] = M @ lres_vals[i,:]
    return interp_lres_vals

def save_stats(output_path: str, variables, means: np.ndarray, maxes: np.ndarray, mins: np.ndarray, stdevs: np.ndarray):
    stats = {}
    stats['mean'] = {}
    stats['maximum'] = {}
    stats['minimum'] = {}
    stats['stdev'] = {}
    for i in range(len(variables)):
        v = variables[i]
        stats['mean'][v] = means[i]
        stats['maximum'][v] = maxes[i]
        stats['minimum'][v] = mins[i]
        stats['stdev'][v] = stdevs[i]
    np.save(output_path, stats)

def main():
    parser = argparse.ArgumentParser(
        description="Computes residuals between an interpolated low-res and high-res dataset",
        usage="python compute_residuals.py --lres_dataset --hres_dataset --interpolation_matrix --start_date --end_date",
    )
    parser.add_argument(
        "--lres_dataset",
        type=str,
        required=True,
        help="Source grid file name (format : grid_[source_grid_name].npz)",
    )
    parser.add_argument(
        "--hres_dataset",
        type=str,
        required=True,
        help="Target grid file name (format : grid_[target_grid_name].npz)",
    )
    parser.add_argument(
        "--interpolation_matrix",
        type=str,
        default=".",
        help="Where the weights file can be found (default : where the script in run).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Where to save stats file.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=".",
        help="Start date/time of training set",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=".",
        help="End date/time of training set",
    )

    args = parser.parse_args()

    compute_residuals(
        args.lres_dataset,
        args.hres_dataset,
        args.interpolation_matrix,
        args.output_path,
        np.datetime64(args.start_date),
        np.datetime64(args.end_date)
    )

if __name__ == "__main__":
    main()
