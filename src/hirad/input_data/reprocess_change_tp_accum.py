import logging
import os
import sys

import torch
import numpy as np

# Reprocess ERA-interpolated data to exclude the tp variable.

# 6H data is all channels, but with 6h accumulation
DATA_SOURCE_6H = "/capstor/scratch/cscs/mmcgloho/datasets/processed/era5-cosmo-1h-all-channels/era-interpolated"
STATS_FILEPATH_6H = "/capstor/scratch/cscs/mmcgloho/datasets/processed/era5-cosmo-1h-all-channels/info"
# 1h data is the updated
DATA_SOURCE_1H = "/capstor/store/cscs/swissai/a161/era5-cosmo-1h-linear-interpolation/train/era-interpolated-with-copernicus-tp/"
STATS_FILEPATH_1H = "/capstor/store/cscs/swissai/a161/era5-cosmo-1h-linear-interpolation/train/info"
OUTPUT_DIR = "/iopsstor/scratch/cscs/mmcgloho/run-1_4/train/era-interpolated"
OUTPUT_STATS_FILEPATH = "/iopsstor/scratch/cscs/mmcgloho/run-1_4/train/info/"
TP_INDEX_6H = 34 # in era-all.yaml
TP_INDEX_1H = 12 # in era.yaml

def process(input_directory_6h: str, input_directory_1h: str, output_directory: str):
    input_6h_filepath = os.path.join(input_directory_6h)
    files = os.listdir(input_6h_filepath)
    files.sort()
    for f in range(len(files)):
        if f % 100 == 0:
            logging.info(f)
        input_1h_file = os.path.join(input_directory_1h, files[f])
        input_6h_file = os.path.join(input_directory_6h, files[f])
        outfile = os.path.join(output_directory, files[f])
        in_data_6h = torch.load(input_6h_file, weights_only=False)
        in_data_1h = torch.load(input_1h_file, weights_only=False)
        in_data_6h[TP_INDEX_6H,:] = in_data_1h[TP_INDEX_1H,:]
        torch.save(in_data_6h, outfile)

def edit_info(info_6h_filepath: str, info_1h_filepath: str, output_filepath: str):
    stats_6h = torch.load(os.path.join(info_6h_filepath, 'era-stats'), weights_only=False)
    stats_1h = torch.load(os.path.join(info_1h_filepath, 'era-stats'), weights_only=False)
    logging.info(f'6h stats: {stats_6h}')
    logging.info(f'1h stats: {stats_1h}')
    for k in stats_6h.keys():
        logging.info(k, stats_6h[k])
        tmp = stats_6h[k]
        tmp[TP_INDEX_6H] = stats_1h[k][TP_INDEX_1H]
        stats_6h[k] = tmp
    logging.info(stats_6h)
    torch.save(stats_6h, os.path.join(output_filepath, 'era-stats'))

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    process(DATA_SOURCE_6H, DATA_SOURCE_1H, OUTPUT_DIR)
    edit_info(STATS_FILEPATH_6H, STATS_FILEPATH_1H, OUTPUT_STATS_FILEPATH)

if __name__ == "__main__":
    main()
