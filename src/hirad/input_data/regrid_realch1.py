
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
from hirad.input_data import interpolate_basic
import yaml
import torch
from pandas import to_datetime

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

TRIM_EDGE = 41
XARRAY_BATCH = 4

# Take anemoi dataset and provide xarray dataarrays for a set of variables.
# returns: list of xarray dataarrays
def anemoi_to_xarray(anemoi_data: Dataset, start_date_index=-1, end_date_index=-1):
	if start_date_index == -1:
		start_date_index = 0
	if end_date_index == -1:
		end_date_index = len(anemoi_data.dates)
	lon = anemoi_data.longitudes
	lat = anemoi_data.latitudes
	eps = [0]  # deterministic
	time = anemoi_data.dates[start_date_index:end_date_index]
	metadata = getMetadataFromOGD()
	dataarrays = []
	variables = anemoi_data.variables
	for var_index in range(anemoi_data.shape[1]):
		logging.info(f'building xarray for {variables[var_index]}')
		ds = xr.Dataset(
			data_vars=dict(
				variable=(["time", "eps", "cell"],
			  np.array(anemoi_data[start_date_index:end_date_index,var_index,:,:])),
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
 
def interpolate_anemoi_range_to_rotlatlon(i_start: int, i_end: int, ds: Dataset, ds_name: str, input_grid: np.ndarray, output_grid: np.ndarray, output_data_path: str, format='torch', output_plots_path: str = None, plot_indices=[0]):
	torch_data = np.zeros([i_end-i_start, len(ds.variables), 1, output_grid.shape[0]])
		
	xarrays = anemoi_to_xarray(ds, i_start, i_end)
	for j in range(len(xarrays)):
		logging.info(f'regridding {ds.variables[j]} for time {ds.dates[i_start]} to {ds.dates[i_end-1]}')
		xarray = xarrays[j]
		start = datetime.datetime.now()
		regridded=regrid.icon2rotlatlon(xarray)
		end = datetime.datetime.now()
		logging.info(f'   regridding took {end-start} seconds')
		torch_data[0:i_end-i_start,j,:,:] = regridded_to_numpy(regridded, trim_edge=TRIM_EDGE)
		
	logging.info('saving torch data')
	for k in range(torch_data.shape[0]):
		interpolate_basic.save_datetime_file(torch_data[k,:], ds.dates[i_start + k], output_data_path, format=format)
		if (i_start + k) in plot_indices:
			logging.info(f'plotting {i_start+k}')
			datestr = interpolate_basic.format_date(ds.dates[i_start+k])
			for v in range(torch_data.shape[1]):
				interpolate_basic.plot_and_save_projection(input_grid[:,0], input_grid[:,1],
							ds[i_start+k,v,0,:],
							os.path.join(output_plots_path, f'{datestr}-{ds.variables[v]}-iconnative.png'),
							s=0.005)
				interpolate_basic.plot_and_save_projection(output_grid[:,0], output_grid[:,1],
							torch_data[k,v,0,:],
							os.path.join(output_plots_path, f'{datestr}-{ds.variables[v]}-rotlatlon.png'),
							s=0.005)

def interpolate_anemoi_to_rotlatlon(infile_anemoi: str, ds_name: str, output_grid: np.ndarray, output_path: str, format='torch', plot_indices=[0]):
	
	# Copy the realch1.yml file to the info directory
	shutil.copy(infile_anemoi, os.path.join(output_path, 'info'))

	with open(infile_anemoi) as realch1_file:
		realch1_config = yaml.safe_load(realch1_file)
	realch1 = open_dataset(realch1_config)
	variables = realch1.variables
	input_grid = np.column_stack((realch1.longitudes, realch1.latitudes))

	# Get the lat/lon info by regridding one variable
	xarrays = anemoi_to_xarray(realch1, 0, 1)
	regridded=regrid.icon2rotlatlon(xarrays[0])
	logging.info('getting geo coords')
	lats, lons = get_geo_coords(regridded, trim_edge=TRIM_EDGE)
	output_grid=np.column_stack((lons, lats))
	
	# Save grid to file
	grid = np.column_stack((lats, lons))
	torch.save(grid, os.path.join(output_path, 'info', 'realch1-lat-lon'))

	# Save stats
	interpolate_basic.save_anemoi_stats(realch1, os.path.join(output_path, f'info/{ds_name}-stats'))

	output_data_path = os.path.join(output_path, ds_name)
	output_plots_path = os.path.join(output_path, 'plots')

	# Split regridding into batches; too many time points seems to not scale well.
	for i in range(0, len(realch1.dates), XARRAY_BATCH):
		start_index = i
		end_index = min(i+XARRAY_BATCH, len(realch1.dates))
		logging.info(f'start={start_index} end={end_index}')
		interpolate_anemoi_range_to_rotlatlon(start_index, end_index, realch1, ds_name,
										input_grid, output_grid, output_data_path, format, output_plots_path, plot_indices)
		

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
	
	interpolate_anemoi_to_rotlatlon(realch1_config_file, 'realch1', None, output_directory, 'numpy', [0])



if __name__ == "__main__":
    main()