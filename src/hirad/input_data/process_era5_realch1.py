import logging
import os
import sys

import numpy as np
import torch

import interpolate_basic

def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call process_era5_realch1.py [era.yaml] [cosmo.yaml] [output directory]')
    infile_era = sys.argv[1]
    infile_realch1 = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)

    erashortname = infile_era.split('/')[-1].split('.')[0]

    logging.basicConfig(
        filename=os.path.join(output_path, f'process-era5-realch1-{erashortname}.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

    logging.info(f'running {sys.argv}')

    format='numpy'
    plot_indices=[0]
    
    interpolate_basic.save_anemoi_as_format(infile_realch1, 'realch1', output_path, plot_indices=plot_indices, format=format)
    

if __name__ == "__main__":
    main()
