import logging
import os
import sys

import datetime
import torch
import numpy as np


from hirad.eval.plotting import plot_map_precipitation, plot_scores_vs_t

def load_all_data(filepath: str):
    files = os.listdir(filepath)
    example = torch.load(os.path.join(filepath, files[0]), weights_only=False)
    dims = (len(files),) + example.shape
    data = np.zeros(dims)
    for f in range(100):
    #for f in range(len(files)):
        if f % 100 == 0:
            logging.info(f)
        curr = torch.load(os.path.join(filepath, files[f]), weights_only=False)
        data[f,:] = curr
    return data

def count_nans(data: np.array):
    nans = np.count_nonzero(np.isnan(data))
    return nans
    
def make_stats(filepath: str):
    data = load_all_data(filepath)
    stats = {}
    num_channels = data.shape[1]
    stats['mean'] = np.zeros(num_channels)
    stats['stdev'] = np.zeros(num_channels)
    stats['minimum'] = np.zeros(num_channels)
    stats['maximum'] = np.zeros(num_channels)
    for k in range(num_channels):
        logging.info(f'channel {k}')
        stats['mean'][k] = np.mean(data[:,k,:,:])
        stats['minimum'][k] = np.min(data[:,k,:,:])
        stats['maximum'][k] = np.max(data[:,k,:,:])
        stats['stdev'][k] = np.std(data[:,k,:,:])
    return stats

def main():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    input_directory = sys.argv[1]
    
    logging.info(f'checking input directory {input_directory}')

    missing_data = []
    corrupt_data = []
    nan_data = []
    check_for_nans = False
    check_for_corrupt = False



    files = os.listdir(input_directory)
    files.sort()
    start_date = datetime.datetime.strptime(files[0],'%Y%m%d-%H%M')
    next_date = datetime.datetime.strptime(files[1],'%Y%m%d-%H%M')
    delta = next_date - start_date
    prev_date = start_date - delta
    
    for f in files:
        curr_date = datetime.datetime.strptime(f,'%Y%m%d-%H%M')
        if curr_date - prev_date != delta:
            logging.info(f'missing data: {prev_date} and {curr_date} not {delta} apart')
            expected_date = prev_date + delta
            while (expected_date < curr_date):
                missing_data.append(datetime.datetime.strftime(expected_date, '%Y%m%d-%H%M'))
                expected_date = expected_date + delta
        if check_for_corrupt:
            try:
                data = torch.load(os.path.join(input_directory, f), weights_only=False)
            except:
                logging.info(f'corrupt data: {curr_date}')
                corrupt_data.append(curr_date)
            if check_for_nans or curr_date == start_date:
                if count_nans(data):
                    logging.info(f'data nans: {curr_date}')
                    nan_data.append(curr_date)
        prev_date = curr_date
    logging.info(f'missing data size {len(missing_data)}: {missing_data}')
    logging.info(f'corrupt data size {len(corrupt_data)}: {corrupt_data}')
    if check_for_nans:
        logging.info(f'nan data: {nan_data}')

if __name__ == "__main__":
    main()
