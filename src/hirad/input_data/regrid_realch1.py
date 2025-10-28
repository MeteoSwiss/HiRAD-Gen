

import logging
import os
import sys

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import numpy as np
from meteodatalab.operators import regrid
import xarray as xr
from meteodatalab import ogd_api
from hirad.input_data.interpolate_basic import plot_and_save_projection
import yaml

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from earthkit.geo.rotate import unrotate

# Take anemoi dataset and provide xarray dataarrays for a set of variables.
# returns: list of xarray dataarrays
def anemoi_to_xarray(anemoi_data: Dataset):
	lon = anemoi_data.longitudes
	lat = anemoi_data.latitudes
	eps = [0]  # deterministic
	time = generate_times(anemoi_data) # anemoi_data.dates?
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

# Get array of times from the anemoi dataset
def generate_times(anemoi_data: Dataset):
    times = []
    curr_time = anemoi_data.start_date.item()
    while curr_time <= anemoi_data.end_date:
        times.append(curr_time)
        curr_time = curr_time + anemoi_data.frequency
    return times

# get the geo coordinates for the rotated lat/lon dataset.
# returns np.array of lats and array of lons
def get_geo_coords(regridded_data: xr.Dataset):
	xmin = regridded_data.metadata.get("longitudeOfFirstGridPointInDegrees")
	xmax = regridded_data.metadata.get("longitudeOfLastGridPointInDegrees")
	dx = regridded_data.metadata.get("iDirectionIncrementInDegrees")
	ymin = regridded_data.metadata.get("latitudeOfFirstGridPointInDegrees")
	ymax = regridded_data.metadata.get("latitudeOfLastGridPointInDegrees")
	dy = regridded_data.metadata.get("jDirectionIncrementInDegrees")
	y = np.arange(ymin,ymax+dy,dy)
	x = np.arange(xmin,xmax+dx,dx)
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

def main():
	# yml format
	realch1_config_file = sys.argv[1]
	output_directory = sys.argv[2]
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	for subdir in ['info', 'plots', 'realch1']:
		if not os.path.exists(os.path.join(output_directory, subdir)):
			os.mkdir(os.path.join(output_directory, subdir))

	with open(realch1_config_file) as realch1_file:
		realch1_config = yaml.safe_load(realch1_file)
	realch1 = open_dataset(realch1_config)
	variables = realch1.variables

	logging.basicConfig(level=logging.INFO)

	xarrays = anemoi_to_xarray(realch1)
	
	# Get the lat/lon info by regridding first variable
	regridded=regrid.icon2rotlatlon(xarrays[0])
	lats, lons = get_geo_coords(regridded)
	logging.info(regridded)
	logging.info(regridded.data)
	logging.info(regridded.data.shape)
	# TODO: Save lat/lon info 
	
	# regridded is in shape (eps, time, variable, x, y)
	# want this in shape (time,channel,ensemble,grid)
	torch_data = np.zeros([len(realch1.dates), len(realch1.variables), 1, len(lats)])
	torch_data[:,0,:,:] = regridded.data.reshape(regridded.shape[1], regridded.shape[0], regridded.shape[3]*regridded.shape[4])
	
	for i in range(1, len(xarrays)):
		xarray = xarrays[i]
		regridded=regrid.icon2rotlatlon(xarray)
		torch_data[:,i,:,:] = regridded.data.reshape(regridded.shape[1], regridded.shape[0], regridded.shape[3]*regridded.shape[4])
	
	# TODO: output each time point into torch file

	# Output plots
	for i in range(torch_data.shape[1]):
		plot_and_save_projection(realch1.longitudes, realch1.latitudes,
							realch1[0,i,0,:], f'{variables[i]}-icon.png', s=0.005)
		plot_and_save_projection(lons, lats,
								torch_data[0,i,0,:], f'{variables[i]}-regridded.png', s=0.005)


if __name__ == "__main__":
    main()