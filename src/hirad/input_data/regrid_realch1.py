
import datetime
import logging
import os
import shutil
import sys

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import numpy as np
from meteodatalab.operators import regrid
import xarray as xr
from meteodatalab import ogd_api
from hirad.input_data.interpolate_basic import plot_and_save_projection
import yaml
import torch
from pandas import to_datetime

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from earthkit.geo.rotate import unrotate

TRIM_EDGE = 41

# Take anemoi dataset and provide xarray dataarrays for a set of variables.
# returns: list of xarray dataarrays
def anemoi_to_xarray(anemoi_data: Dataset):
	lon = anemoi_data.longitudes
	lat = anemoi_data.latitudes
	eps = [0]  # deterministic
	time = anemoi_data.dates
	metadata = getMetadataFromOGD()
	dataarrays = []
	variables = anemoi_data.variables
	for var_index in range(anemoi_data.shape[1]):
		ds = xr.Dataset(
			data_vars=dict(
				variable=(["time", "eps", "cell"], np.array(anemoi_data[:,var_index,:,:])),
			),
			coords=dict(
				eps=eps,
				time=time,
				lon=("cell", lon),
				lat=("cell", lat),
			),
			attrs=dict(description=f'xarray from anemoi dataset for {variables[var_index]}',
				metadata=metadata),
		)
		dataarrays.append(ds.to_dataarray())
	return dataarrays

# Run a request to get the metadata, so that we can fake out an xarray.
def getMetadataFromOGD():
    lead_times = ["P0DT0H"]
    req = ogd_api.Request(
		collection="ogd-forecasting-icon-ch1",
		variable="TOT_PREC", #assuming this won't cause problems; we're only using grid info
		ref_time="latest",
		perturbed=False,
		lead_time=lead_times,
    )	
    tot_prec = ogd_api.get_from_ogd(req)
    return tot_prec.metadata

# get the geo coordinates for the rotated lat/lon dataset.
# returns np.array of lats and array of lons
def get_geo_coords(regridded_data: xr.Dataset, trim_edge=0):
	xmin = regridded_data.metadata.get("longitudeOfFirstGridPointInDegrees")
	xmax = regridded_data.metadata.get("longitudeOfLastGridPointInDegrees")
	dx = regridded_data.metadata.get("iDirectionIncrementInDegrees")
	ymin = regridded_data.metadata.get("latitudeOfFirstGridPointInDegrees")
	ymax = regridded_data.metadata.get("latitudeOfLastGridPointInDegrees")
	dy = regridded_data.metadata.get("jDirectionIncrementInDegrees")
	y = np.arange(ymin,ymax+dy,dy)
	x = np.arange(xmin,xmax+dx,dx)
	# trim x and y according to trim_edge.
	# (Have manually verified that when doing this, the outputs are the same as
	# trimming post-projection)
	y = y[trim_edge:len(y)-trim_edge]
	x = x[trim_edge:len(x)-trim_edge]
	sp_lat = regridded_data.metadata.get("latitudeOfSouthernPoleInDegrees") # -43.0. north_pole_lat = 43.0
	sp_lon = regridded_data.metadata.get("longitudeOfSouthernPoleInDegrees") # 10.0. north_pole_lon = 190.0
	xcoords = np.meshgrid(x,y)[0].flatten()
	ycoords = np.meshgrid(x,y)[1].flatten()
	# Expect south pole rotation of lon=10, latitude=-43
	logging.info(f'sp_lat = {sp_lat}, sp_lon = {sp_lon}')
	rotated_crs = ccrs.RotatedPole(
    	pole_longitude=(sp_lon + 180) % 360, pole_latitude=sp_lat * -1  # 190, 43
	)
    # Project onto PlateCarree. Geodetic produces similar coordinates (within 10 nanometers)
	dst_grid = ccrs.PlateCarree()
	geo_coords = dst_grid.transform_points(rotated_crs, xcoords, ycoords)
	lats = geo_coords[:,1]
	lons = geo_coords[:,0]
	return lats, lons

def regridded_to_numpy(regridded: xr.DataArray, trim_edge=0):
	# regridded is in shape (eps, time, variable, x, y)
	# want this in shape (time, channel, ensemble, grid)
	# First, trim the edge
	data = regridded.data[:,:,:,
					   trim_edge:regridded.data.shape[3]-trim_edge,
					   trim_edge:regridded.data.shape[4]-trim_edge]
	# reshape to (time,channel,ensemble,grid)
	data = data.reshape(data.shape[1], data.shape[0], data.shape[3]*data.shape[4])
	return data

def main():
	# yml format
	realch1_config_file = sys.argv[1]
	output_directory = sys.argv[2]
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	for subdir in ['info', 'plots', 'realch1']:
		if not os.path.exists(os.path.join(output_directory, subdir)):
			os.mkdir(os.path.join(output_directory, subdir))

	logging.basicConfig(
	    filename=os.path.join(output_directory, 'regrid_realch1.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

	# Copy the realch1.yml file to the info directory
	shutil.copy(realch1_config_file, os.path.join(output_directory, 'info'))

	with open(realch1_config_file) as realch1_file:
		realch1_config = yaml.safe_load(realch1_file)
	realch1 = open_dataset(realch1_config)
	variables = realch1.variables

	xarrays = anemoi_to_xarray(realch1)
	
	# Get the lat/lon info by regridding first variable
	logging.info(f'regridding {variables[0]} for time {realch1.start_date} to {realch1.end_date}')
	start = datetime.datetime.now()
	regridded=regrid.icon2rotlatlon(xarrays[0])
	end = datetime.datetime.now()
	logging.info(f'   regridding took {end-start} seconds')
	logging.info('getting geo coords')
	lats, lons = get_geo_coords(regridded, trim_edge=TRIM_EDGE)
	
	# Save lat/lon info
	grid = np.column_stack((lats, lons))
	torch.save(grid, os.path.join(output_directory, 'info', 'realch1-lat-lon'))
	
	# regridded is in shape (eps, time, variable, x, y)
	# want this in shape (time,channel,ensemble,grid)
	# nervous about the reshaping screwing things up, but that's why we plot the interpolated data to visually check.
	torch_data = np.zeros([len(realch1.dates), len(realch1.variables), 1, len(lats)])
	torch_data[:,0,:,:] = regridded_to_numpy(regridded, trim_edge=TRIM_EDGE)
	
	for i in range(1, len(xarrays)):
		logging.info(f'regridding {variables[i]} for time {realch1.start_date} to {realch1.end_date}')
		xarray = xarrays[i]
		start = datetime.datetime.now()
		regridded=regrid.icon2rotlatlon(xarray)
		end = datetime.datetime.now()
		logging.info(f'   regridding took {end-start} seconds')
		torch_data[:,i,:,:] = regridded_to_numpy(regridded, trim_edge=TRIM_EDGE)
	
	# Output each time point into torch file
	logging.info('saving torch data')
	for t in range(torch_data.shape[0]):	
		fmtdate = to_datetime(realch1.dates[t]).strftime('%Y%m%d-%H%M')
		torch.save(torch_data[t,:], os.path.join(output_directory, 'realch1', fmtdate))

	# Output plots for each variable, for first time point
	for i in range(torch_data.shape[1]):
		plot_and_save_projection(realch1.longitudes, realch1.latitudes,
					realch1[0,i,0,:],
					os.path.join(output_directory, 'plots', f'{variables[i]}-iconnative.png'),
					s=0.005)
		plot_and_save_projection(lons, lats,
					torch_data[0,i,0,:],
					os.path.join(output_directory, 'plots', f'{variables[i]}-rotlatlon.png'),
					s=0.005)


if __name__ == "__main__":
    main()