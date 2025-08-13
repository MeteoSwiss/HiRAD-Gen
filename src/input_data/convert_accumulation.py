import logging
import os

import numpy as np
import torch

TP_INDEX = 12
INFILE_PATH = '/store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-linear-interpolation-full/era/'
NULL_CONST = -1

# Get list of files, sorted by date
def era_infiles():
    f = os.listdir(INFILE_PATH)
    f.sort()
    return f

# Load data for a single time point
def load_data(t: str):
    data = torch.load(os.path.join(INFILE_PATH, t), weights_only=False)
    return data[TP_INDEX, 0, :]

# Load all total precipitation into one matrix of size (timesteps, gridpoints)
# Input: timesteps. If None, will get all timestamps.
def load_all_tp(timesteps:int | None):
    logging.info('loading all tp')
    files = era_infiles()
    if not timesteps:
        timesteps = len(files)
    grid = load_data(files[0])
    gridsize = len(grid)
    accum_tp = np.zeros((timesteps, gridsize))
    for t in range(timesteps):
        f = files[t]
        if t % 100 == 0: 
            logging.info(t)
        gridtp = load_data(f)
        accum_tp[t,:] = gridtp
    return accum_tp

def interpolate_1h_from_6h(accum_tp: np.array):
    # set up the interpolated time points.
    interpolated_tp = np.ones_like(accum_tp) * NULL_CONST
    # get time point and locations of all zeroes
    zero_t, zero_l = np.where(accum_tp == 0)
    # for each 0, fill all 1-hour time points for the past 6h with 0s
    for i in range(len(zero_t)):
        # avoid array index OOB
        min_ind = max(zero_t[i] - 5, 0)
        interpolated_tp[min_ind:(zero_t[i]+1), zero_l[i]] = 0

    # iterate through grid points
    for j in range(accum_tp.shape[1]):
        # find the first zero (where index > 5, just to simplify OOB issues)
        loc_zeros = zero_t[np.where(zero_l == j)]
        loc_zeros.sort()
        i = 0
        while loc_zeros[i] < 6:
            i = i + 1
        first_zero = loc_zeros[i]

        # work forwards
        for i in range(first_zero+1, interpolated_tp.shape[0]):
            # current-time value is 6h-accumulation minus the previous 5h of 1h-accumulation 
            val = accum_tp[i,j] - np.sum(interpolated_tp[i-5:i,j])

            # check if we already have a 0 where we shouldn't have one
            # this indicates that something doesn't add up
            if val < 0 or (interpolated_tp[i,j] != NULL_CONST and interpolated_tp[i,j] != val):
                logging.warning(f'found a problem at index {i} {j}. val={val} interpolated_tp={interpolated_tp[i,j]}')
                logging.warning(f'accum_tp[i,j] = {accum_tp[i,j]} interpolated_tp[i-5:i,j] = {interpolated_tp[i-5:i,j]}')
                logging.warning(f'accum_tp[i-10:i:10]={accum_tp[i-10:i+10,j]}')
                for k in range(0, min(i+10,  interpolated_tp.shape[0])):
                    logging.warning(f'{k}\t{accum_tp[k,j]:.9f}\t{interpolated_tp[k,j]:.9f}')
                          

            interpolated_tp[i,j] = val

        # work backwards, starting with what should be the last nonzero value
        # before our pivot point
        for i in range(first_zero-6, -1, -1):
            # current time point is accumulation at i+5 (current point is first in moving window)
            # minus the future 5 hours of 1h-accumulation 
            val = accum_tp[i+5,j] - np.sum(interpolated_tp[i+1:i+6,j])
            interpolated_tp[i,j] = val
    return interpolated_tp

# Calculate errors
def check_interpolation(interpolated_tp, accum_tp):
    err = np.zeros_like(interpolated_tp)
    for i in range(5, interpolated_tp.shape[0]):
        accum = np.sum(interpolated_tp[i-5:i+1,:], 0)
        err[i,:] = accum - accum_tp[i,:]
    logging.info(f'num errors are {np.count_nonzero(err)}')
    logging.info(f'sum error is {np.sum(err)}')
    return np.sum(err)

# Find the least-rainy time point
# Not currently in use-- written for purposes of potential later optimization)
def find_driest_day():
    # Make a grid of reference dates
    files = era_infiles()
    grid = load_data(files[0])
    has_zero = np.zeros_like(grid)
    min_rainy_spots = grid.shape[0]
    min_rainy_day = ''
    i = 0
    for f in files:
        if i % 100 == 0: 
            print(f)
        i = i + 1
        gridtp = load_data(f)
        rainy_spots = np.count_nonzero(gridtp)
        if rainy_spots < min_rainy_spots:
            min_rainy_day = f
            min_rainy_spots = rainy_spots
            print(f'min rainy day is now {f} with {rainy_spots}')
            #min rainy day is 20180421-0100 = 0
    print(f'min rainy day is {min_rainy_day}')
    return min_rainy_day


def main():
    accum_tp = load_all_tp(100)
    interpolated_tp = interpolate_1h_from_6h(accum_tp)
    #print(accum_tp[1:100,0])
    #print(interpolated_tp[1:100,0])
    check_interpolation(interpolated_tp, accum_tp)


if __name__ == "__main__":
    main()
