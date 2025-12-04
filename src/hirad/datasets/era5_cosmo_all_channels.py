from .base import DownscalingDataset, ChannelMetadata
import os
import numpy as np
import torch
from typing import List, Tuple
import yaml
import torch.nn.functional as F
import time
# import zarr

from hirad.datasets import era5_cosmo

from hirad.utils.console import PythonLogger

logger = PythonLogger(__name__)


class ERA5_COSMO_ALL_CHANNELS(era5_cosmo.ERA5_COSMO):
    def __init__(self, 
                dataset_path: str, 
                input_channel_names: List[str] = [], 
                output_channel_names: List[str] = [], 
                static_channel_names: List[str] = [], 
                transform_channels: List[str] = [],
                n_month_hour_channels: int = None,
                ):
        super().__init__()

        # Reuse all paths except for input data path, and info
        self._era5_path = os.path.join(dataset_path, 'era-copernicus-interpolated')
        self._info_path = os.path.join(dataset_path, 'info')

        # inherit file list from cosmo

        # Load cosmo info and channel names
        with open(os.path.join(self._info_path,'cosmo.yaml'), 'r') as file:
            self._cosmo_info = yaml.safe_load(file)
            if output_channel_names:
                self._cosmo_indeces = [self._cosmo_info['select'].index(name) for name in output_channel_names]
            else:
                self._cosmo_indeces = list(range(len(self._cosmo_info['select'])))
                output_channel_names = self._cosmo_info['select']
            self._cosmo_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                        else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                        for name in self._cosmo_info['select'] if name in output_channel_names]

        # Load era5 info and channel names
        with open(os.path.join(self._info_path,'era.yaml'), 'r') as file:
            self._era_info = yaml.safe_load(file)
            if input_channel_names:
                self._era_indeces = [self._era_info['select'].index(name) for name in input_channel_names]
            else:
                self._era_indeces = list(range(len(self._era_info['select'])))
                input_channel_names = self._era_info['select']
            self._era_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                    else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                    for name in self._era_info['select'] if name in input_channel_names]
        
        # Load stats for normalizing channels of input and output

        cosmo_stats = torch.load(os.path.join(self._info_path,'cosmo-stats'), weights_only=False)
        self.output_mean = cosmo_stats['mean'][self._cosmo_indeces]
        self.output_std = cosmo_stats['stdev'][self._cosmo_indeces]

        era_stats = torch.load(os.path.join(self._info_path,'era-stats'), weights_only=False)
        self.input_mean = era_stats['mean'][self._era_indeces]
        self.input_std = era_stats['stdev'][self._era_indeces]
        if self.static_data is not None:
            self.input_mean = np.concatenate((self.input_mean, self.static_mean), axis=0)
            self.input_std = np.concatenate((self.input_std, self.static_std), axis=0)

        # FEATURE: load the mean and std values for transformed channels and update the normalization statistics
    
        self.input_transforms = {}
        self.input_inverse_transforms = {}
        self.output_transforms = {}
        self.output_inverse_transforms = {}
        for transform_descriptor in transform_channels:
            channel, transformation = transform_descriptor.split('-')
            input_channel_idx = self._era_info['select'].index(channel) if channel in self._era_info['select'] else None
            output_channel_idx = self._cosmo_info['select'].index(channel) if channel in self._cosmo_info['select'] else None
            if transformation.startswith('box_cox'):
                lmbda_str = transformation.split('_')[-1]
                lmbda = float(transformation.split('_')[-1])/(10**(len(lmbda_str)-1))
                print(f"Applying Box-Cox transformation with lambda={lmbda} to channel {channel} (input idx: {input_channel_idx}, output idx: {output_channel_idx})")
                if input_channel_idx is not None:
                    self.input_transforms[input_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_transform(x, lmbda)
                    self.input_inverse_transforms[input_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_inverse_transform(x, lmbda)
                    self.input_mean[input_channel_idx] = torch.load(os.path.join(self._info_path,f"era5-{transform_descriptor}-mean"), weights_only=False)
                    self.input_std[input_channel_idx] = torch.load(os.path.join(self._info_path,f"era5-{transform_descriptor}-std"), weights_only=False)
                if output_channel_idx is not None:
                    self.output_transforms[output_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_transform(x, lmbda)
                    self.output_inverse_transforms[output_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_inverse_transform(x, lmbda)
                    self.output_mean[output_channel_idx] = torch.load(os.path.join(self._info_path,f"cosmo-{transform_descriptor}-mean"), weights_only=False)
                    self.output_std[output_channel_idx] = torch.load(os.path.join(self._info_path,f"cosmo-{transform_descriptor}-std"), weights_only=False)
            else:
                raise ValueError(f"Transformation: {transformation} for channel {channel} not implemented.")

    def __getitem__(self, idx):
        """Get cosmo and era5 interpolated to cosmo grid"""
        # get data point
        # squeeze the ensemble dimesnsion
        # reshape to image_shape
        # flip so that it starts in top-left corner (by default it is bottom left)
        # orig_shape = [350,542] #TODO currently padding to be divisible by 16
        orig_shape = self.image_shape()
        try:
            era5_data = torch.load(os.path.join(self._era5_path,self._file_list[idx]), weights_only=False)[self._era_indeces]
        except:
            logger.error(f"Error loading file {os.path.join(self._era5_path,self._file_list[idx])}")
            raise
        era5_data = np.flip(era5_data \
                                .squeeze() \
                                .reshape(-1,*orig_shape),
                            1)
        era5_data = np.concatenate((era5_data, self.static_data), axis=0) if self.static_data is not None else era5_data
        era5_data = self.normalize_input(era5_data)

        try:
            cosmo_data = torch.load(os.path.join(self._cosmo_path,self._file_list[idx]), weights_only=False)[self._cosmo_indeces]
        except:
            logger.error(f"Error loading file {os.path.join(self._cosmo_path,self._file_list[idx])}")
            raise
        cosmo_data = np.flip(cosmo_data\
                                .squeeze() \
                                .reshape(-1,*orig_shape),
                            1)
        cosmo_data = self.normalize_output(cosmo_data)

        if self._n_month_hour_channels is not None and self._n_month_hour_channels>0:
            # extract month and hour from filename
            filename = self._file_list[idx]
            date_str, hour_str = filename.split('-')
            month = int(date_str[4:6])
            hour = int(hour_str[0:2])

            time_grid = self.make_time_grids(hour, month)
            era5_data = np.concatenate((era5_data, time_grid), axis=0)