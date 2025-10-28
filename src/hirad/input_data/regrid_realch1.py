

import logging

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import numpy as np
from meteodatalab.operators import regrid
import xarray as xr
from meteodatalab import ogd_api
from hirad.input_data.interpolate_basic import plot_and_save_projection

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from earthkit.geo.rotate import unrotate

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

def get_geo_coords(regridded_data: xr.Dataset, sp_lat=None, sp_lon=None):
	xmin = regridded_data.metadata.get("longitudeOfFirstGridPointInDegrees")
	xmax = regridded_data.metadata.get("longitudeOfLastGridPointInDegrees")
	dx = regridded_data.metadata.get("iDirectionIncrementInDegrees")
	ymin = regridded_data.metadata.get("latitudeOfFirstGridPointInDegrees")
	ymax = regridded_data.metadata.get("latitudeOfLastGridPointInDegrees")
	dy = regridded_data.metadata.get("jDirectionIncrementInDegrees")
	y = np.arange(ymin,ymax+dy,dy)
	x = np.arange(xmin,xmax+dx,dx)
	# TODO, this parameter is not producing what I want it to.
	if not sp_lat or not sp_lon:
		sp_lat = regridded_data.metadata.get("latitudeOfSouthernPoleInDegrees") # -43.0. north_pole_lat = 43.0
		sp_lon = regridded_data.metadata.get("longitudeOfSouthernPoleInDegrees") # 10.0. north_pole_lon = 190.0
	xcoords = np.meshgrid(x,y)[0].flatten()
	ycoords = np.meshgrid(x,y)[1].flatten()
	logging.info(f'sp_lat = {sp_lat}, sp_lon = {sp_lon}')
	#geo_coords = unrotate(ycoords, xcoords, sp_lat*-1, sp_lon-180)
	#geo_coords = unrotate(ycoords, xcoords, 43, 190)
	#geo_coords = unrotate(ycoords, xcoords, -43, 190) # close
	#geo_coords = unrotate(ycoords, xcoords, -43, 10)  # produces range: 26W-8W, 41S-51.5S. image correct orientation.
	geo_coords = unrotate(ycoords, xcoords, sp_lat, sp_lon)
     
	return geo_coords, ycoords, xcoords

logging.basicConfig(level=logging.INFO)
realch1 = open_dataset('/scratch/mch/fzanetta/data/anemoi/datasets/mch-realch1-fdb-1km-2020-2020-1h-pl13-v0.1.zarr')
var = 'TD_2M'
var_index = realch1.variables.index(var)
myxarray = anemoi_to_xarray(realch1, var).to_dataarray()
regridded=regrid.icon2rotlatlon(myxarray)

plot_and_save_projection(realch1.longitudes, realch1.latitudes,
                         realch1[0,var_index,0,:], f'{var}-icon.png', s=0.01)
plot_and_save_projection(myxarray.lon, myxarray.lat,
                         myxarray[0,0,0,:], f'{var}-xarray.png', s=0.005)

geo_coords, ycoords, xcoords = get_geo_coords(regridded)
logging.info(geo_coords)
# South pole rotation of lon=10, latitude=-43
rotated_crs = ccrs.RotatedPole(
    pole_longitude=190, pole_latitude=43
)
plot_and_save_projection(xcoords, ycoords,
                         regridded[0,0,0,:], f'{var}-regridded-rotated-projection.png',
                        projection=rotated_crs) # picture looks accurate.
plot_and_save_projection(geo_coords[1], geo_coords[0],
                         regridded[0,0,0,:], f'{var}-regridded.png')

# None of these work.
for sp_lat in [-43, 43]:
	for sp_lon in [10, 170, 190, 350]:
		geo_coords, ycoords, xcoords = get_geo_coords(regridded, sp_lat=sp_lat, sp_lon=sp_lon)
		logging.info(geo_coords)
		plot_and_save_projection(geo_coords[1], geo_coords[0],
								regridded[0,0,0,:], f'{var}-regridded_{sp_lat}_{sp_lon}.png')
		plot_and_save_projection(geo_coords[0], geo_coords[1],
								regridded[0,0,0,:], f'{var}-regridded-reversed_{sp_lat}_{sp_lon}.png')