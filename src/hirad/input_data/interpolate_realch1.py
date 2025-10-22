

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

def _read_input(era_config_file: str, realch1_config_file: str, ) -> tuple[Dataset, Dataset, array.array]:
    """
    Read both ERA and REA-L-CH1 data, and return the 2m
    temperature values for the time range under COSMO.
    """
    # trim edge removes boundary, we will use the same 
    with open(realch1_config_file) as realch1_file:
        realch1_config = yaml.safe_load(realch1_file)
    realch1 = open_dataset(realch1_config)
    with open(era_config_file) as era_file:
        era_config = yaml.safe_load(era_file)
    era = open_dataset(era_config)
    # Subset the ERA dataset to have REAL-CH-1 area/dates.
    start_date = realch1.metadata()['start_date']
    end_date = realch1.metadata()['end_date']
    # load era5 2m-temperature in the time-range of cosmo
    # area = N, W, S, E
    min_lat = min(realch1.latitudes) - ERA_MARGIN_DEGREES
    max_lat = max(realch1.latitudes) + ERA_MARGIN_DEGREES
    min_lon = min(realch1.longitudes) - ERA_MARGIN_DEGREES
    max_lon = max(realch1.longitudes) + ERA_MARGIN_DEGREES
    era = open_dataset(era, start=start_date, end=end_date,
                    area=(max_lat, min_lon, min_lat, max_lon))
    

    copernicus_netcdf = []
    for f in COPERNICUS_FILES:
        netcdf_data = netCDF4.Dataset(f)
        copernicus_netcdf.append(netcdf_data)
        
    return (era, realch1, copernicus_netcdf)


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