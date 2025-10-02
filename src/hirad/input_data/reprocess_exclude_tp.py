import logging
import os
import sys

import torch
import numpy as np

# Reprocess ERA-interpolated data to exclude the tp variable.

TP_INDEX = 12

def process(input_directory: str, output_directory: str):
    input_filepath = os.path.join(input_directory, 'era-interpolated')
    files = os.listdir(input_filepath)
    files.sort()
    for f in range(len(files)):
        if f % 100 == 0:
            logging.info(f)
        outfile = os.path.join(output_directory, 'era-interpolated', files[f])
        if (not os.path.exists(outfile)) or (os.path.getsize(outfile) < 26000000):
            in_data = torch.load(os.path.join(input_filepath, files[f]), weights_only=False)
            out_data = in_data[0:TP_INDEX,:]
            torch.save(out_data, outfile)

def edit_info(input_filepath: str, output_filepath: str):
    stats = torch.load(os.path.join(input_filepath, '/info', 'era-stats'), weights_only=False)
    logging.info(stats)
    for k in stats.keys():
        logging.info(k, stats[k])
        stats[k] = stats[k][0:TP_INDEX]
    logging.info(stats)
    torch.save(stats, os.path.join(output_filepath, "/info", "era-stats"))

def main():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    process(input_directory, output_directory)
    #edit_info(input_directory, output_directory)

if __name__ == "__main__":
    main()
