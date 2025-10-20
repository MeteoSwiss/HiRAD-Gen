

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
from meteodatalab import icon_grid
from meteodatalab.operators import regrid
import torch
import multiprocessing
import xarray as xr
from meteodatalab import ogd_api

def anemoi_to_xarray(anemoi_data: Dataset, variable):
    lon = anemoi_data.longitudes
    lat = anemoi_data.latitudes
    eps = [0]  # deterministic
    time = generate_times(anemoi_data)
    var_index = anemoi_data.variables.index(variable)
    metadata = getMetadataFromOGD()
    
    ds = xr.Dataset(
		data_vars=dict(
			variable=(["time", "eps", "cell"], np.array(anemoi_data.data[:,var_index,:,:])),
		),
		coords=dict(
            eps=eps,
			time=time,
			lon=("cell", lon),
			lat=("cell", lat),
		),
		attrs=dict(description=f'xarray from anemoi dataset for {variable}',
             metadata=metadata),
	)
    print(ds)
    return ds

def getMetadataFromOGD():
    lead_times = ["P0DT0H"]
    req = ogd_api.Request(
		collection="ogd-forecasting-icon-ch1",
		variable="TOT_PREC",
		ref_time="latest",
		perturbed=False,
		lead_time=lead_times,
    )	
    tot_prec = ogd_api.get_from_ogd(req)
    return tot_prec.metadata

def generate_times(anemoi_data: Dataset):
    times = []
    curr_time = anemoi_data.start_date.item()
    while curr_time <= anemoi_data.end_date:
        times.append(curr_time)
        curr_time = curr_time + anemoi_data.frequency
    return times
        
    

def get_coeffs_path(model: str):
    return coeffs_path
    # TODO some value error check file avialable ofr sth.

def remap():
    # get UUID for 1-km native grid
    #icon_grid_uuid = get_uuid('icon-ch1-eps')
    
    coeffs_path = f'/store_new/mch/msopr/icon_workflow_2/iconremap-weights/{model}-rotlatlon.nc'
    coeffs = xr.open_dataset(coeffs_path)
    
    indices = coeffs["rbf_B_glbidx"].values
    weights = coeffs["rbf_B_wgt"].values
    geo = {
        "gridType": "rotated_ll",
        "longitudeOfSouthernPoleInDegrees": coeffs.north_pole_lon - 180,
        "latitudeOfSouthernPoleInDegrees": -1 * coeffs.north_pole_lat,
    }
    dst = RegularGrid(
        crs=_get_crs(geo),
        nx=coeffs.nx,
        ny=coeffs.ny,
        xmin=coeffs.xmin,
        ymin=coeffs.ymin,
        xmax=coeffs.xmax,
        ymax=coeffs.ymax,
    )

realch1 = open_dataset('/scratch/mch/fzanetta/data/anemoi/datasets/mch-realch1-fdb-1km-2020-2020-1h-pl13-v0.1.zarr')
myxarray = anemoi_to_xarray(realch1, "TOT_PREC")
regrid.icon2rotlatlon(myxarray)
