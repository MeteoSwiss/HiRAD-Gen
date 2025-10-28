

import logging
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
# returns: list of xarray dataarrays, and list of variable indices (anemoi)
def anemoi_to_xarray(anemoi_data: Dataset, variables):
	lon = anemoi_data.longitudes
	lat = anemoi_data.latitudes
	eps = [0]  # deterministic
	time = generate_times(anemoi_data)
	metadata = getMetadataFromOGD()
	dataarrays = []
	var_indices = []
	for variable in variables:
		var_index = anemoi_data.variables.index(variable)
		var_indices.append(var_index)

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
		dataarrays.append(ds.to_dataarray())
	return dataarrays, var_indices

# Run a request to get the metadata, so that we can fake out an xarray.
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

	with open(realch1_config_file) as realch1_file:
		realch1_config = yaml.safe_load(realch1_file)
	realch1 = open_dataset(realch1_config)

logging.basicConfig(level=logging.INFO)

realch1 = open_dataset('/scratch/mch/fzanetta/data/anemoi/datasets/mch-realch1-fdb-1km-2020-2020-1h-pl13-v0.1.zarr')
variables = ['TD_2M', 'TOT_PREC']
myxarrays, var_indices = anemoi_to_xarray(realch1, variables)

for i in range(len(variables)):
	myxarray = myxarrays[i]
	regridded=regrid.icon2rotlatlon(myxarray)
	plot_and_save_projection(realch1.longitudes, realch1.latitudes,
							realch1[0,var_indices[i],0,:], f'{variables[i]}-icon.png', s=0.005)
	plot_and_save_projection(myxarray.lon, myxarray.lat,
							myxarray[0,0,0,:], f'{variables[i]}-xarray.png', s=0.005)

	lats, lons = get_geo_coords(regridded)

	plot_and_save_projection(lons, lats,
							regridded[0,0,0,:], f'{variables[i]}-regridded.png', s=0.005)


if __name__ == "__main__":
    main()