import logging
import os
import sys

import numpy as np
import torch

import interpolate_basic

def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call process_copenicus_cosmo.py [copernicus.yaml] [cosmo.yaml] [output directory]')
    infile_copernicus = sys.argv[1]
    infile_cosmo = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_path, f'process-copernicus-cosmo.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

    logging.info(f'running {sys.argv}')

    output_grid = None

    if infile_cosmo.endswith('yaml'):
        cosmo = interpolate_basic.read_anemoi_ds(infile_cosmo)
        output_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes))
    else:
        # This must be a lat-lon torch file.
        cosmo_latlon = torch.load(infile_cosmo, weights_only=False)
        lats = cosmo_latlon[:,0]
        lons = cosmo_latlon[:,1]
        output_grid = np.column_stack((lons, lats))

    # interpolate copernicus
    format = 'numpy'
    plot_indices=[0,24,25]
    interpolate_basic.interpolate_netcdf_to_grid(infile_copernicus, 'copernicus', output_grid, output_path=output_path, format=format, plot_indices=plot_indices)


if __name__ == "__main__":
    main()
