import logging
import os
import sys

import numpy as np
import torch

import interpolate_basic

def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call process_era5_cosmo.py [era.yaml] [cosmo.yaml] [output directory]')
    infile_era = sys.argv[1]
    infile_cosmo = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)

    erashortname = infile_era.split('/')[-1].split('.')[0]

    logging.basicConfig(
        filename=os.path.join(output_path, f'process-era5-cosmo-{erashortname}.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

    logging.info(f'running {sys.argv}')
    #output_plots_path = None

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

    # interpolate era
    format = 'numpy'
    plot_indices=[0]
    interpolate_basic.interpolate_anemoi_to_grid(infile_era, 'era', output_grid, output_path=output_path, format=format, plot_indices=plot_indices)
    # save era and cosmo input/output data into same format
    # Save cosmo data
    #if infile_cosmo.endswith('yaml'):
    #    interpolate_basic.save_anemoi_as_format(infile_cosmo, 'cosmo', output_path, plot_indices=plot_indices, format=format)

    # Save ERA data (subsetted)
    lats = output_grid[:,1]
    lons = output_grid[:,0]
    min_lat = min(lats) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lat = max(lats) + interpolate_basic.ERA_MARGIN_DEGREES
    min_lon = min(lons) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lon = max(lons) + interpolate_basic.ERA_MARGIN_DEGREES
    area=(max_lat, min_lon, min_lat, max_lon)
    logging.info(f'projecting onto era area {area}')
    # skip plotting as we did it already in interpolation step
    #interpolate_basic.save_anemoi_as_format(infile_era, 'era', output_path, plot_indices=plot_indices, format=format, area=area,
    #                                     start_date=cosmo.start_date, end_date=cosmo.end_date)
    #interpolate_basic.save_anemoi_as_format(infile_cosmo, 'cosmo', output_path, plot_indices=plot_indices, format=format)
    

if __name__ == "__main__":
    main()
