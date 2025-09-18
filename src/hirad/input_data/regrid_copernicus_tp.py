import logging
import netCDF4
import xarray
import numpy as np
import torch
import datetime
from scipy.interpolate import griddata

from hirad.eval.plotting import plot_map_precipitation, plot_scores_vs_t
from hirad.eval.metrics import absolute_error

import interpolate_basic

import sys
from pathlib import Path

import os
print (os.getcwd())

sys.path.insert(0, Path(__file__).parent.as_posix())


CDF_FILENAME_BALFRIN = "/store_new/mch/msopr/hirad-gen/copernicus-datasets/tp-janfeb2020.nc"
#CDF_FILENAME_CLARIDEN_TP = "/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2019-2020.nc"
#CDF_FILENAME_CLARIDEN_TP = "/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2017-2018-n320.nc"
CDF_FILENAME_CLARIDEN_TP = "/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2015-2016.nc"


BASE_FILEPATH = "/capstor/store/"
INPUT_DATA_FILEPATH = "mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-linear-interpolation-full/"
OUTPUT_DATA_FILEPATH_ERA_INTERPOLATED = "/capstor/store/cscs/swissai/a161/era5-cosmo-1h-linear-interpolation/train/era-interpolated-with-copernicus-tp"
OUTPUT_DATA_FILEPATH_ERA = "/capstor/store/cscs/swissai/a161/era5-cosmo-1h-linear-interpolation/train/era-with-copernicus-tp"
TP_INDEX = 12

LAT = np.arange(-4.42, 3.36 + 0.02, 0.02)
LON = np.arange(-6.82, 4.80 + 0.02, 0.02)
RELAX_ZONE = 19 # Number of points dropped on each side (relaxation zone)

def extract_grib_values(grib_data):
    grib_lat = grib_data['latitude'][:]
    grib_lon = grib_data['longitude'][:]
    grib_t2m = grib_data['t2m'][:]

def extract_lat_lon_025(data):
    logging.info('extracting lat/lon')
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

def extract_lat_lon_n320(data):
    lat = data['latitudes'][:]
    lon = data['longitudes'][:]
    logging.info('extracting lat/lon')
    logging.info(f'lat lon shapes {lat.shape} {lon.shape}')

def extract_values(data, variable, area=None):
    values = data[variable][:]
    print(values.shape)
    if area:
        lat = data['latitude'][:]
        lon = data['longitude'][:]
        # https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        latli = np.argmin( np.abs(lat - area[2]))
        latui = np.argmin( np.abs(lat - area[0]))
        lonli = np.argmin( np.abs(lon - area[1]))
        lonui = np.argmin( np.abs(lon - area[3]))
        lat = data['latitude'][latli:latui]
        lon = data['longitude'][lonli:lonui]
        values = data[variable][latli:latui,lonli:lonui]
    return np.reshape(values, (values.shape[0], values.shape[1]*values.shape[2]))
 
def reshape_to_cosmo(vals):
    return vals.reshape((len(LAT)-RELAX_ZONE*2, len(LON)-RELAX_ZONE*2))

def calc_errors(cosmo1, era1):
    make_plots = True

    prev_netcdf_regrid = []

    netcdf_error = np.zeros(cosmo1.dates.shape)
    era_norm_error = np.zeros(cosmo1.dates.shape)
    netcdf_early_error = np.zeros(cosmo1.dates.shape)
    netcdf_late_error = np.zeros(cosmo1.dates.shape)
    
    output_grid= np.column_stack((cosmo1.longitudes, cosmo1.latitudes))

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

def process_era_interpolated(netcdf_data, netcdf_tp_values, input_data_filepath, output_interpolated_filepath, netcdf_grid, cosmo_grid):
    make_plots = False
    #for t in range(100):
    for t in range(netcdf_tp_values.shape[0]):
        netcdf_date = netcdf_data['valid_time'][t]
        date_filename = datetime.datetime.fromtimestamp(netcdf_date, datetime.UTC).strftime('%Y%m%d-%H%M')
        t1 = datetime.datetime.now()
        era_filename = os.path.join(input_data_filepath, "era-interpolated", date_filename)
        output_filename = os.path.join(output_interpolated_filepath, date_filename)
        if os.path.exists(era_filename):
            requires_processing = False
            if date_filename == '20200615-2100':
                requires_processing = True
            #if os.path.exists(output_filename):
                # test the output to make sure it is not corrupted.
                #requires_processing = False
                #try:
                #    torch.load(output_filename, weights_only=False)
                #except:
                #    requires_processing = True
            if requires_processing:
                era_data = torch.load(era_filename, weights_only=False)
                t2 = datetime.datetime.now()
                #if t % 100 == 0:
                logging.info(f'regridding {date_filename} (netcdf date: {netcdf_date})')
                interpolated_tp = griddata(netcdf_grid, netcdf_tp_values[t,:], cosmo_grid, method='linear')
                t3 = datetime.datetime.now()
                if make_plots and max > 0.0002:
                    nans = np.count_nonzero(np.isnan(interpolated_tp))
                    nonzeros = np.count_nonzero(interpolated_tp)
                    max = np.max(interpolated_tp)            
                    logging.info(f'nonzeros: {nonzeros} nans: {nans} max: {max}')
                    if max > 0.0002:
                        cosmo_filename = os.path.join(input_data_filepath, "cosmo", date_filename)
                        cosmo_data = torch.load(cosmo_filename, weights_only=False)
                        plot_map_precipitation(reshape_to_cosmo(interpolated_tp), f'plots/tp/{date_filename}-netcdf-regrid')
                        # 3 is tp index in cosmo data
                        plot_map_precipitation(reshape_to_cosmo(cosmo_data[3,:]), f'plots/tp/{date_filename}-cosmo')
                        plot_map_precipitation(reshape_to_cosmo(era_data[TP_INDEX,0,:]/6), f'plots/tp/{date_filename}-era-norm')
                era_data[TP_INDEX,0,:] = interpolated_tp
                torch.save(era_data, output_filename)
                t4 = datetime.datetime.now()

def process_era(netcdf_data, netcdf_tp_values):
    for t in range(netcdf_tp_values.shape[0]):
        netcdf_date = netcdf_data['valid_time'][t]
        date_filename = datetime.datetime.fromtimestamp(netcdf_date, datetime.UTC).strftime('%Y%m%d-%H%M')
        era_filename = os.path.join(BASE_FILEPATH, INPUT_DATA_FILEPATH, "era", date_filename)
        if os.path.exists(era_filename):
            era_data = torch.load(era_filename, weights_only=False)
            t2 = datetime.datetime.now()
            logging.info(f'regridding {date_filename} (netcdf date: {netcdf_date})')
            interpolated_tp = griddata(netcdf_grid, netcdf_tp_values[t,:], era_grid, method='linear')
            t3 = datetime.datetime.now()
            era_data[TP_INDEX,0,:] = interpolated_tp
            torch.save(era_data, os.path.join(OUTPUT_DATA_FILEPATH_ERA, date_filename))
            t4 = datetime.datetime.now()

def make_stats():
    #cosmo_files = os.listdir(os.path.join(BASE_FILEPATH, INPUT_DATA_FILEPATH, 'cosmo'))
    #era_files = os.listdir(os.path.join(BASE_FILEPATH, INPUT_DATA_FILEPATH, 'cosmo'))
    
    stats = torch.load(os.path.join(BASE_FILEPATH, INPUT_DATA_FILEPATH, 'info', 'era-stats'), weights_only=False)
    print(stats)
    set1 = netCDF4.Dataset("/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2015-2016.nc")
    set2 = netCDF4.Dataset("/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2017-2018.nc")
    set3 = netCDF4.Dataset("/capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2019-2020.nc")
    set1_tp = extract_values(set1, 'tp')
    set2_tp = extract_values(set2, 'tp')
    set3_tp = extract_values(set3, 'tp')
    all_tp = np.row_stack((set1_tp, set2_tp, set3_tp))
    print(all_tp.shape)
    all_tp = all_tp.reshape(all_tp.shape[0] * all_tp.shape[1], 1)
    mean = np.mean(all_tp)
    max = np.max(all_tp)
    min = np.min(all_tp)
    stdev = np.std(all_tp)
    stats['mean'][TP_INDEX] = mean
    stats['maximum'][TP_INDEX] = max
    stats['minimum'][TP_INDEX] = min
    stats['stdev'][TP_INDEX] = stdev
    print(stats)
    torch.save(stats, os.path.join(OUTPUT_DATA_FILEPATH_ERA_INTERPOLATED, 'era-stats'))


#process_era(netcdf_data, netcdf_tp_values)


def main():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    logging.info('loading data')
    netcdf_file = sys.argv[1]
    input_data_filepath = sys.argv[2]
    output_interpolated_filepath = sys.argv[3]

    netcdf_data = netCDF4.Dataset(netcdf_file)
    logging.info(netcdf_data)

    logging.info('processing netcdf data')
    netcdf_latitudes, netcdf_longitudes = extract_lat_lon_025(netcdf_data)
    netcdf_grid=np.column_stack((netcdf_longitudes, netcdf_latitudes))
    cosmo_grid = torch.load(os.path.join(input_data_filepath, 'info/cosmo-lat-lon'), weights_only=False)
    cosmo_grid = np.column_stack((cosmo_grid[:,1], cosmo_grid[:,0]))
    logging.info(f'netcdf grid shape {netcdf_grid.shape}')
    logging.info(f'{netcdf_grid[1:10,:]}')
    logging.info(f'cosmo grid shape {cosmo_grid.shape}')
    logging.info(f'{cosmo_grid[1:10,:]}')

    netcdf_tp_values = extract_values(netcdf_data, 'tp')

    
    process_era_interpolated(netcdf_data, netcdf_tp_values, input_data_filepath, output_interpolated_filepath, netcdf_grid, cosmo_grid)
    

if __name__ == "__main__":
    main()

