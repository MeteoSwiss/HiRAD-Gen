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
from meteodatalab.operators import regrid
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


def regrid_all(era: Dataset, realch1: Dataset, copernicus: array.array):
    # iterate through the dates
    realch1 = regrid_realch1
    # convert to xarray.dataarray
    regrid.icon2rotlatlon

    pass

def regrid_realch1():
    # Use the meteodatalab functions to regrid the realch1 anemoi data (one time point)
    # onto the rotated lat lon
    # save the output as torch
    # return the np array
    pass

def regrid_era():
    # Take the output grid from realch1-regrid (rotated lat lon).
    # regrid all variables *except* tp directly from era5 data
    # regrid the pt variable from the netcdf data
    # save the output as torch
    pass

def main():
    # read REA-L-CH1 latlon grid
    era_config_file = sys.argv[1]
    realch1_latlon_file = sys.argv[2]
    netcdf_file = sys.argv[3]
    output_directory = sys.argv[4]

    realch1_grid = torch.load(realch1_latlon_file, weights_only=False)
    # read ERA input
    min_lat = min(realch1_grid[:,0]) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lat = min(realch1_grid[:,0]) + interpolate_basic.ERA_MARGIN_DEGREES
    min_lon = min(realch1_grid[:,1]) - interpolate_basic.ERA_MARGIN_DEGREES
    max_lon = min(realch1_grid[:,1]) + interpolate_basic.ERA_MARGIN_DEGREES
    era = interpolate_basic.read_era5_anemoi(era_config_file,
                                             area=(max_lat, min_lon, min_lat, max_lon))
    era_grid = np.column_stack((era.longitudes, era.latitudes))
    
    # read copernicus input for tp variable
    netcdf_data = netCDF4.Dataset(netcdf_file)
    logging.info('processing netcdf data')
    netcdf_latitudes, netcdf_longitudes = regrid_copernicus_tp.extract_lat_lon_025(netcdf_data)
    netcdf_grid=np.column_stack((netcdf_longitudes, netcdf_latitudes))
    # TODO: start and end date functionality
    netcdf_tp_values = regrid_copernicus_tp.extract_values(netcdf_data, 'tp', start_date=era.start_date, end_date=era.end_date)
    assert(netcdf_tp_values.shape[0] == era.shape[0])


    # Iterate over ERA time range, which should be subsetted in configuration.
    for i in range(era.shape[0]):
        t = era.dates[i]
        # Get everything but the tp variable
        tp_index = era.variables.index('tp')
        logging.info('tp index' + tp_index)
        # shape time, channel, ensemble, grid
        era_for_time = np.delete(era[i,:,:,:], tp_index, axis=0)
        era_regridded = interpolate_basic.regrid(era_for_time, era_grid, realch1_grid)
        copernicus_regridded = interpolate_basic.regrid(netcdf_data[0,:], netcdf_grid, realch1_grid)
        output=np.stack((era_regridded, copernicus_regridded), axis=1)
        filename = os.path.join(output_directory, 'era-copernicus-interpolated',
                                interpolate_basic._format_date(t))
        torch.save(output, filename)

    return 0