import hirad.input_data.interpolate_basic as interpolate_basic
import hirad.input_data.regrid_copernicus_tp as regrid_copernicus_tp

import datetime
import logging
import os
import shutil
import sys
import yaml
import array

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import netCDF4
import numpy as np
from pandas import to_datetime
from scipy.interpolate import griddata
import torch
import multiprocessing
import xarray

# Margin to use for ERA dataset (to avoid nans from interpolation at boundary)
ERA_MARGIN_DEGREES = 1.0
COPERNICUS_FILES = ['/store_new/mch/msopr/hirad-gen/copernicus-datasets/tp-2015-2016.nc',
                    '/store_new/mch/msopr/hirad-gen/copernicus-datasets/tp-2017-2018.nc',
                    '/store_new/mch/msopr/hirad-gen/copernicus-datasets/tp-2019-2020.nc']

def _read_input(era_config_file: str, realch1_latlon_file: str) -> tuple[Dataset, Dataset, array.array, np.ndarray]:
    """
    Read ERA data, and return the values for the area under REA-L-CH1 (plus a margin).
    """
    # read the lat/lon data for REA-L-CH1
    realch1_latlon = torch.load(realch1_latlon_file)
    # we expect the start and end dates to be specified in config.
    with open(era_config_file) as era_file:
        era_config = yaml.safe_load(era_file)
    era = open_dataset(era_config)
    # Subset the ERA dataset to have REAL-CH-1 area.
    # area = N, W, S, E
    min_lat = min(realch1_latlon[:,0]) - ERA_MARGIN_DEGREES
    max_lat = max(realch1_latlon[:,0]) + ERA_MARGIN_DEGREES
    min_lon = min(realch1_latlon[:,1]) - ERA_MARGIN_DEGREES
    max_lon = max(realch1_latlon[:,1]) + ERA_MARGIN_DEGREES
    era = open_dataset(era,
                    area=(max_lat, min_lon, min_lat, max_lon))
    copernicus_netcdf = []
    for f in COPERNICUS_FILES:
        netcdf_data = netCDF4.Dataset(f)
        copernicus_netcdf.append(netcdf_data)
    return (era, copernicus_netcdf, realch1_latlon)


def main():
    # read REA-L-CH1 latlon grid
    era_config_file = sys.argv[1]
    realch1_latlon_file = sys.argv[2]
    netcdf_file = sys.argv[3]
    output_directory = sys.argv[4]

    logging.basicConfig(
        filename=os.path.join(output_directory, 'interpolate_realch1.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 
    
    # Copy ERA yml file
    shutil.copy(era_config_file, os.path.join(output_directory, 'info', 'era.yaml'))

    logging.info('reading realch1 lat/lon')
    realch1_latlon = torch.load(realch1_latlon_file, weights_only=False)
    realch1_lat = realch1_latlon[:,0]
    realch1_lon = realch1_latlon[:,1]
    # read ERA input
    min_lat = min(realch1_lat) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lat = max(realch1_lat) + interpolate_basic.ERA_MARGIN_DEGREES
    min_lon = min(realch1_lon) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lon = max(realch1_lon) + interpolate_basic.ERA_MARGIN_DEGREES
    logging.info('reading era')
    
    era = interpolate_basic.read_anemoi_ds(era_config_file,
                                             area=(max_lat, min_lon, min_lat, max_lon))
    era_grid = np.column_stack((era.longitudes, era.latitudes))
    realch1_grid = np.column_stack((realch1_lon, realch1_lat))
    logging.info(f'lat lon area is {min_lat}-{max_lat} {min_lon}-{max_lon}')

    # save era stats and lat lon
    interpolate_basic.save_anemoi_latlon_grid(era, os.path.join(output_directory, 'info', 'era-lat-lon'))
    interpolate_basic.save_anemoi_stats(era, os.path.join(output_directory, 'info', 'era-stats'))
    
    # read copernicus input for tp variable
    logging.info('reading copernicus')
    netcdf_data = netCDF4.Dataset(netcdf_file)
    logging.info('processing netcdf data')
    netcdf_latitudes, netcdf_longitudes = regrid_copernicus_tp.extract_lat_lon_025(netcdf_data)
    netcdf_grid=np.column_stack((netcdf_longitudes, netcdf_latitudes))

    # TODO: start and end date functionality
    netcdf_tp_values = regrid_copernicus_tp.extract_values(netcdf_data, 'tp', start_date=era.start_date, end_date=era.end_date)
    assert(netcdf_tp_values.shape[0] == era.shape[0])
    # todo: incorporate this somehow
    netcdf_tp_values = netcdf_tp_values.reshape((netcdf_tp_values.shape[0], 1,1, netcdf_tp_values.shape[1]))

    # save copernicus stats and lat lon
    torch.save(np.column_stack((netcdf_grid[:,1], netcdf_grid[:,0])),
               os.path.join(output_directory, 'info', 'copernicus-lat-lon'))
    regrid_copernicus_tp.make_stats(os.path.join(output_directory, 'info'),
                                    os.path.join(output_directory, 'info'),
                                    netcdf_tp_values)

    # Iterate over ERA time range, which should be subsetted in configuration.
    tp_index = era.variables.index('tp')
    logging.info(f'tp index {tp_index}')

    plot_indices = {12}

    logging.info('interpolating')
    #for i in plot_indices:
    for i in range(era.shape[0]):
        # T
        t = era.dates[i]
        # Get everything but the tp variable
        era_for_time = era[i,:,:,:]
        era_regridded = interpolate_basic.regrid(era_for_time, era_grid, realch1_grid)
        # Regrid TP from copernicus
        copernicus_regridded = interpolate_basic.regrid(netcdf_tp_values[i,:], netcdf_grid, realch1_grid)
        # Concatenate and save
        era_regridded[tp_index,:] = copernicus_regridded
        #output=np.concatenate((era_regridded, copernicus_regridded), axis=0)
        datefmt = interpolate_basic._format_date(t)
        filename = os.path.join(output_directory, 'era-copernicus-interpolated',
                                datefmt)
        torch.save(era_regridded, filename)

        if i in plot_indices:
            realch1var = ['t2m', '10u', '10v', 'tp']
            realch1_data = torch.load(os.path.join(output_directory, 'realch1', datefmt), weights_only=False)
            for j in range(realch1_data.shape[0]):
                interpolate_basic.plot_and_save_projection(realch1_lon, realch1_lat, realch1_data[j,:],
                                                            os.path.join(output_directory, 'plots',
                                                                        f'{datefmt}-{realch1var[j]}-realch1'))
            for j in range(era_regridded.shape[0]):
                interpolate_basic.plot_and_save_projection(era.longitudes, era.latitudes, era_for_time[j,:],
                                                           os.path.join(output_directory, 'plots',
                                                                        f'{datefmt}-{era.variables[j]}-era'))
                interpolate_basic.plot_and_save_projection(realch1_lon, realch1_lat,
                                                           era_regridded[j,0,:],
                                                           os.path.join(output_directory, 'plots',
                                                                        f'{datefmt}-{era.variables[j]}-interpolated'))
    return 0

if __name__ == "__main__":
    main()
