import logging
import netCDF4
import xarray
from anemoi.datasets import open_dataset
import numpy as np
import yaml

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

from hirad.eval.plotting import plot_map_precipitation, plot_scores_vs_t
from hirad.eval.metrics import compute_mae, absolute_error


import interpolate_basic

import sys
from pathlib import Path

import os
print (os.getcwd())

sys.path.insert(0, Path(__file__).parent.as_posix())

ANEMOI_1H_FILENAME = "/scratch/mch/omiralle/anemoi/aifs-ea-an-oper-0001-mars-n320-2015-2020-1h-v1-with-ERA51.zarr"
ANEMOI_6H_FILENAME = "/scratch/mch/apennino/data/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr"
COSMO_6H_FILENAME = "/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr"
COSMO_1H_FILENAME = "/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-1h-v3-pl13.zarr"
COSMO_CONFIG_FILE="src/input_data/cosmo.yaml"
CDF_FILENAME = "/store_new/mch/msopr/hirad-gen/copernicus-datasets/tp-janfeb2020.nc"
GRIB_FILENAME = "/store_new/mch/msopr/hirad-gen/copernicus-datasets/surface-janfeb2020.grib"


LAT = np.arange(-4.42, 3.36 + 0.02, 0.02)
LON = np.arange(-6.82, 4.80 + 0.02, 0.02)
RELAX_ZONE = 19 # Number of points dropped on each side (relaxation zone)

def extract_grib_values(grib_data):
    grib_lat = grib_data['latitude'][:]
    grib_lon = grib_data['longitude'][:]
    grib_t2m = grib_data['t2m'][:]

def extract_lat_lon(data):
    lat = data['latitude'][:]
    lon = data['longitude'][:]
    output_lat = np.zeros(len(lat)* len(lon))
    output_lon = np.zeros(len(lat) * len(lon))
    for i in range(len(lat)):
        if i % 10 == 0:
            print(i)
        for j in range(len(lon)):
           grid_index = i * len(lon) + j
           output_lat[grid_index] = lat[i]
           output_lon[grid_index] = lon[j]
    return output_lat, output_lon

def extract_values(data, variable):
    values = data[variable][:]
    return np.reshape(values, (values.shape[0], values.shape[1]*values.shape[2]))

def extract_netcdf_values(netcdf_data):
    netcdf_lat = netcdf_data['latitude'][:]
    netcdf_lon = netcdf_data['longitude'][:]
    netcdf_tp = netcdf_data['tp'][:,:]
    logging.info(f'num nonzeros in tp is {np.count_nonzero(netcdf_tp)}')
    values = np.zeros((netcdf_tp.shape[0], netcdf_tp.shape[1]*netcdf_tp.shape[2]))
    latitudes = np.zeros(values.shape[1])
    longitudes = np.zeros(values.shape[1])
    # You could probably get this by reshaping, but I can't be bothered.
    for i in range(len(netcdf_lat)):
        if i % 10 == 0:
            print(i)
        for j in range(len(netcdf_lon)):
           grid_index = i * len(netcdf_lon) + j
           values[:,grid_index] = netcdf_tp[:,i,j]
           latitudes[grid_index] = netcdf_lat[i]
           longitudes[grid_index] = netcdf_lon[j]
    reshape_values = np.reshape(netcdf_tp, (netcdf_tp.shape[0], netcdf_tp.shape[1]*netcdf_tp.shape[2]))
    logging.info(f'Array equal? {np.array_equal(values, reshape_values)}')
    return reshape_values, latitudes, longitudes
 
def reshape_to_cosmo(vals):
    return vals.reshape((len(LAT)-RELAX_ZONE*2, len(LON)-RELAX_ZONE*2))

root = logging.getLogger()
root.setLevel(logging.INFO)

logging.info('loading data')
grib_data = xarray.load_dataset(GRIB_FILENAME)
netcdf_data = netCDF4.Dataset(CDF_FILENAME)
cosmo1 = open_dataset(COSMO_1H_FILENAME, trim_edge=19, select=['tp'],start='2016-01-01',end='2016-02-29')
cosmo6 = open_dataset(COSMO_6H_FILENAME, trim_edge=19, select=['tp'],start='2016-01-01',end='2016-02-29')
era1 = open_dataset(ANEMOI_1H_FILENAME, select=['tp'],start='2016-01-01',end='2016-02-29')
era6 = open_dataset(ANEMOI_6H_FILENAME, select=['tp'],start='2016-01-01',end='2016-02-29')
logging.info('loading data complete')

output_grid= np.column_stack((cosmo1.longitudes, cosmo1.latitudes))

logging.info('processing netcdf data')
netcdf_latitudes, netcdf_longitudes = extract_lat_lon(netcdf_data)
netcdf_values = extract_values(netcdf_data, 'tp')
#netcdf_values, netcdf_latitudes, netcdf_longitudes = extract_netcdf_values(netcdf_data=netcdf_data)
netcdf_grid=np.column_stack((netcdf_longitudes, netcdf_latitudes))

make_plots = True

prev_netcdf_regrid = []

netcdf_error = np.zeros(cosmo1.dates.shape)
era_norm_error = np.zeros(cosmo1.dates.shape)
netcdf_early_error = np.zeros(cosmo1.dates.shape)
netcdf_late_error = np.zeros(cosmo1.dates.shape)

for t in range(4):
#for t in range(len(cosmo1.dates)):
    date = cosmo1.dates[t]
    era_date = era1.dates[t]
    if date != era_date:
        logging.error('dates do not match: cosmo date: {date}, era date: {era_date}')
    if date != netcdf_data['valid_time'][t]:  
        logging.error(f'dates do not match: cosmo date: {date}, netcdf: {netcdf_data["valid_time"][t]}')


    # plot cosmo
    if make_plots:
        plot_map_precipitation(values=reshape_to_cosmo(cosmo1[t,:]), filename=f'plots/tp/{date}-cosmo1')
        if t % 6 == 0:
            if cosmo6.dates[t//6] != date:
                logging.error(f'dates do not match: cosmo1: {date}, cosmo6: {cosmo6.dates[t//6]}')
            plot_map_precipitation(values=reshape_to_cosmo(cosmo6[t//6,:]), filename=f'plots/tp/{date}-cosmo6')
    
    # plot netcdf
    netcdf_vals = netcdf_values[t,:].reshape((1,1,netcdf_values.shape[1]))
    netcdf_regrid=interpolate_basic.regrid(netcdf_vals, netcdf_grid, output_grid)
    if make_plots:
        plot_map_precipitation(reshape_to_cosmo(netcdf_regrid), f'plots/tp/{date}-netcdf-refactor')

    # plot era
    era_grid = np.column_stack((era1.longitudes, era1.latitudes))
    era1_regrid = interpolate_basic.regrid(era1[t,:], era_grid, output_grid)
    if make_plots:
        plot_map_precipitation(reshape_to_cosmo(era1_regrid/6), f'plots/tp/{date}-era1-norm')

        #if t % 6 == 0:
        #    if era6.dates[t//6] != date:
        #        logging.error(f'dates do not match: era1: {date}, era6: {era6.dates[t//6]}')
        #    era6_regrid = interpolate_basic.regrid(era6[t//6,:], era_grid, output_grid)
        #    plot_map_precipitation(reshape_to_cosmo(era6_regrid), f'plots/tp/{date}-era6')

    era_norm_error[t] = np.mean(absolute_error(era1_regrid/6, cosmo1[t,:]))
    netcdf_error[t] = np.mean(absolute_error(netcdf_regrid, cosmo1[t,:]))
    logging.info(f'era norm error: {era_norm_error[t]} netcdf err: {netcdf_error[t]}')
    if t>0:
        netcdf_early_error[t] = np.mean(absolute_error(prev_netcdf_regrid, cosmo1[t,:]))
        netcdf_late_error[t-1] = np.mean(absolute_error(netcdf_regrid, cosmo1[t-1,:]))
        logging.info(f'netcdf early err: {netcdf_early_error[t]}, netcdf late err: {netcdf_late_error[t-1]}')
    prev_netcdf_regrid = netcdf_regrid

maes = {}
maes['era normalized'] = era_norm_error
maes['copernicus'] = netcdf_error
maes['copernicus-early'] = netcdf_early_error
maes['copernicus-late'] = netcdf_late_error
plot_scores_vs_t(maes, times=cosmo1.dates, filename='plots/errors.png')

#fig = plt.figure()
#fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
#interpolate_basic.plot_projection(ax, longitudes=cosmo1.longitudes, latitudes=cosmo1.latitudes, values=cosmo1[0,:])
#fig.savefig('cosmo1.png')

#fig = plt.figure()
#fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
#interpolate_basic.plot_projection(ax, longitudes=cosmo1.longitudes, latitudes=cosmo1.latitudes, values=cosmo6[0,:])
#fig.savefig('cosmo6.png')








