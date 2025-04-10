from anemoi.datasets import open_dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from scipy.interpolate import griddata

COSMO_PATH = '/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr' # path in Balfrin
ERA_PATH = '/scratch/mch/apennino/data/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr' # path in Balfrin

# trim edge removes boundary
cosmo = open_dataset(COSMO_PATH, select="2t", trim_edge=20) # open cosmo, select only 2m-temperature
start_date = cosmo.metadata()['start_date'] # get start and end date of cosmo
end_date = cosmo.metadata()['end_date']
era = open_dataset(ERA_PATH, select="2t", start=start_date, end=end_date) # load era5 2m-temperature in the time-range of cosmo


# get indeces of era5 data that is in the bounding rectangle of cosmo data - this is just for plotting
min_lat_cosmo = min(cosmo.latitudes)
max_lat_cosmo = max(cosmo.latitudes)
min_lon_cosmo = min(cosmo.longitudes)
max_lon_cosmo = max(cosmo.longitudes)
box_lat = np.logical_and(era.latitudes>=min_lat_cosmo,era.latitudes<=max_lat_cosmo)
box_lon = np.logical_and(era.longitudes>=min_lon_cosmo,era.longitudes<=max_lon_cosmo)
indeces = np.where(box_lon*box_lat)


#### Approach 1 #########################################################
#### Scipy Interpolate ##################################################

grid = np.column_stack((era.longitudes, era.latitudes)) # stack lon-lat columns of era5 points
values = np.array(era[0,0,0,:]) # get era grid 2m-temperature values on the first avaialble date-time

interp_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes)) # stack lon-lat column of cosmo points

values_int = griddata(grid,values,interp_grid,method='linear') # interpolate era5 to cosmo grid using scipy griddata linear


################ plotting ################################################

# plot era original
fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
p = ax.scatter(x=era.longitudes[indeces], y=era.latitudes[indeces], c=era[0, 0, 0, :][indeces])
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.colorbar(p, label="K", orientation="horizontal")
plt.savefig("temperature-2m-era.jpg")

# plot cosmo original
fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
p = ax.scatter(x=cosmo.longitudes, y=cosmo.latitudes, c=cosmo[0, 0, 0, :])
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.colorbar(p, label="K", orientation="horizontal")
plt.savefig("temperature-2m-cosmo.jpg")

#plot inerpolated era5
fig = plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
p = ax.scatter(x=cosmo.longitudes, y=cosmo.latitudes, c=values_int)
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.colorbar(p, label="K", orientation="horizontal")
plt.savefig("temperature-2m-era-downscaled.jpg")