from .base import DownscalingDataset, ChannelMetadata
import os
import numpy as np
import torch
from typing import List, Tuple
import yaml
import torch.nn.functional as F
import time
# import zarr

from hirad.utils.console import PythonLogger

logger = PythonLogger(__name__)

DATASET_ORIG_PATH = '/capstor/store/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-linear-interpolation-full'

class ERA5_COSMO(DownscalingDataset):
    def __init__(self, 
                dataset_path: str, 
                input_channel_names: List[str] = [], 
                output_channel_names: List[str] = [], 
                static_channel_names: List[str] = [], 
                transform_channels: List[str] = [],
                n_month_hour_channels: int = None,
                ):
        super().__init__()

        #TODO switch hanbdling paths to Path rather than pure strings
        self._n_month_hour_channels = n_month_hour_channels
        self._dataset_path = dataset_path
        self._era5_path = os.path.join(dataset_path, 'era-interpolated')
        self._cosmo_path = os.path.join(dataset_path, 'cosmo')
        self._info_path = os.path.join(DATASET_ORIG_PATH, 'info')
        # self._static_path = '/capstor/store/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-linear-interpolation-full/static'# os.path.join(dataset_path, 'static')
        self._static_path = os.path.join(DATASET_ORIG_PATH, 'static')# os.path.join(dataset_path, 'static')
        # self._zarr_path = os.path.join(dataset_path, 'dataset.zarr')

        # load file list (each file is one date-time state)
        self._file_list = sorted(os.listdir(self._cosmo_path))

        # open zarr store
        # self._zarr_store = zarr.open(self._zarr_path, mode='r')
        # self.era5 = self._zarr_store['era5']
        # self.cosmo = self._zarr_store['cosmo']

        # Load static info and channel names
        if static_channel_names:
            with open(os.path.join(self._static_path, 'cosmo-static.yaml'), 'r') as file:
                self._static_info = yaml.safe_load(file)
                self._static_indeces = [self._static_info['select'].index(name) for name in static_channel_names]
                self._static_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                        else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                        for name in self._static_info['select'] if name in static_channel_names]
            static_data = torch.load(os.path.join(self._static_path,'cosmo-static'), weights_only=False)[self._static_indeces]
            orig_shape = self.image_shape()
            self.static_data = np.flip(static_data \
                                    .squeeze() \
                                    .reshape(-1,*orig_shape),
                                1)
            self.static_mean = self.static_data.mean(axis=(1,2))
            self.static_std = self.static_data.std(axis=(1,2))
        else:
            self.static_data = None

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

        return torch.tensor(cosmo_data),\
                torch.tensor(era5_data)

    def __len__(self):
        return len(self._file_list)


    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        lat_lon = torch.load(os.path.join(self._info_path,'cosmo-lat-lon'), weights_only=False)
        return lat_lon[:,1]


    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        lat_lon = torch.load(os.path.join(self._info_path,'cosmo-lat-lon'), weights_only=False)
        return lat_lon[:,0]


    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        channels = self._era_channels + self._static_channels if self.static_data is not None else self._era_channels
        if self._n_month_hour_channels is not None and self._n_month_hour_channels>0:
            for i in range(self._n_month_hour_channels):
                channels.append(ChannelMetadata("hour-enc",f"{i}"))
            for i in range(self._n_month_hour_channels):
                channels.append(ChannelMetadata("month-enc",f"{i}"))
        return channels

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return self._cosmo_channels


    def time(self) -> List:
        """Get time values from the dataset."""
        #TODO Choose the time format and convert to that, currently it's a string from a filename
        return [file.split('.')[0] for file in self._file_list]


    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        #TODO load from info, I hardcode it for now (cosmo from anemoi-datasets minus trim-edge=20)
        return 352,544 #TODO 350,542 is orig size, UNet requires dimenions divisible by 16, for now, I just add zeros to orig images
    

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        for channel_idx, transform in self.input_transforms.items():
            x[channel_idx,::] = transform(x[channel_idx,::])
        return (x - self.input_mean.reshape((self.input_mean.shape[0],1,1))) \
                / self.input_std.reshape((self.input_std.shape[0],1,1))


    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        if self._n_month_hour_channels is not None and self._n_month_hour_channels>0:
            x = x[:,:-2*self._n_month_hour_channels,:,:]
        x = x * self.input_std.reshape((self.input_std.shape[0],1,1)) \
                + self.input_mean.reshape((self.input_mean.shape[0],1,1))
        for channel_idx, inverse_transform in self.input_inverse_transforms.items():
            x[:,channel_idx,::] = inverse_transform(x[:,channel_idx,::])
        return x


    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        for channel_idx, transform in self.output_transforms.items():
            x[channel_idx,::] = transform(x[channel_idx,::])
        return (x - self.output_mean.reshape((self.output_mean.shape[0],1,1))) \
                / self.output_std.reshape((self.output_std.shape[0],1,1))


    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        x = x * self.output_std.reshape((self.output_std.shape[0],1,1)) \
                + self.output_mean.reshape((self.output_mean.shape[0],1,1))
        for channel_idx, inverse_transform in self.output_inverse_transforms.items():
            x[:,channel_idx,::] = inverse_transform(x[:,channel_idx,::])
        return x

    def box_cox_transform(self, channel_array: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply Box-Cox transformation to the data."""
        channel_array = np.clip(channel_array, 0, None)
        return (np.power(channel_array, lmbda) - 1) / lmbda

    def box_cox_inverse_transform(self, channel_array: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply inverse Box-Cox transformation to the data."""
        channel_array = np.clip(channel_array, -1/lmbda, None)
        return np.power((lmbda * channel_array) + 1, 1 / lmbda)

    def make_time_grids(self, hour, month):
        """
        Create multi-frequency cyclic sin/cos feature grids for hour and month.

        Parameters
        ----------
        hour : int
            Hour of day, 0-23
        month : int
            Month of year, 1-12

        Returns
        -------
        grid : np.ndarray, shape (C, H, W)
            Channels = [sin(k*hour), cos(k*hour), sin(k*month), cos(k*month) for each k frequency]
        """
        H, W = self.image_shape()
        hour_freqs = np.arange(1, self._n_month_hour_channels//2 + 1)
        month_freqs = np.arange(1, self._n_month_hour_channels//2 + 1)

        channels = []

        # --- hour encodings ---
        for k in hour_freqs:
            angle = 2 * np.pi * k * (hour % 24) / 24.0
            channels.append(np.sin(angle))
            channels.append(np.cos(angle))

        # --- month encodings ---
        for k in month_freqs:
            angle = 2 * np.pi * k * ((month - 1) % 12) / 12.0
            channels.append(np.sin(angle))
            channels.append(np.cos(angle))

        channels = np.array(channels, dtype=np.float32)
        grid = np.tile(channels[:, None, None], (1, H, W))  # (C, H, W)

        return grid