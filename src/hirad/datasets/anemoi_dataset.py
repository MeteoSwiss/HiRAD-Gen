from .base import DownscalingDataset, ChannelMetadata

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import datetime
import os
import numpy as np
from pandas import to_datetime
import torch
from typing import List, Tuple
import yaml
import torch.nn.functional as F
import time
# import zarr

from .constants import REAL_TO_ERA_CHANNEL_MAP, ERA_TO_REAL_CHANNEL_MAP
from hirad.utils.console import PythonLogger
from hirad.utils.dataset_utils import GridData


logger = PythonLogger(__name__)

# Margin to use for ERA dataset (to avoid nans from interpolation at boundary)
INPUT_MARGIN_DEGREES = 0.5

class AnemoiDataset(DownscalingDataset):
    def __init__(self,
                type: str,
                input_anemoi_dataset_path: str,
                target_anemoi_dataset_path: str,
                corrected_tp_path: str,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None,
                input_channel_names: List[str] = [], 
                output_channel_names: List[str] = [], 
                static_channel_names: List[str] = [], 
                transform_channels: List[str] = [],
                transform_input_means: dict = {},
                transform_input_stdevs: dict = {},
                transform_output_means: dict = {},
                transform_output_stdevs: dict = {},
                n_month_hour_channels: int = None,
                trim_edge: int = 0,
                ):
        super().__init__()

        input_dataset = type.split('_')[1]
        target_dataset = type.split('_')[2]
        real_target = target_dataset == 'real'

        if input_dataset != 'era5':
            raise ValueError(f"Input dataset {input_dataset} not supported for AnemoiDataset. Only 'era5' is supported.")
        if target_dataset != 'cosmo' and target_dataset !='real':
            raise ValueError(f"Target dataset {target_dataset} not supported for AnemoiDataset. Only 'cosmo' and 'real' are supported.")

        if real_target:
            # Map output channel names from real to era5
            output_channel_names_real = [ERA_TO_REAL_CHANNEL_MAP[name] for name in output_channel_names]

        self._corrected_tp_path = corrected_tp_path
        #TODO switch hanbdling paths to Path rather than pure strings
        self._n_month_hour_channels = n_month_hour_channels
        if start_date is not None and end_date is not None:
            assert start_date < end_date, "start_date must be before end_date"
            self._output_dataset = open_dataset(target_anemoi_dataset_path, select=output_channel_names_real if real_target else output_channel_names, start=start_date, end=end_date, trim_edge=trim_edge)
        else:
            self._output_dataset = open_dataset(target_anemoi_dataset_path, select=output_channel_names_real if real_target else output_channel_names, trim_edge=trim_edge)
        assert self._output_dataset.shape[1] == len(output_channel_names)

        # Load ERA dataset, trimming the area and limiting the dates to the target dataset
        start_date = self._output_dataset.metadata()['start_date'] if start_date is None else start_date
        end_date = self._output_dataset.metadata()['end_date'] if end_date is None else end_date
        min_lat = min(self._output_dataset.latitudes) - INPUT_MARGIN_DEGREES
        max_lat = max(self._output_dataset.latitudes) + INPUT_MARGIN_DEGREES
        min_lon = max(0, min(self._output_dataset.longitudes) - INPUT_MARGIN_DEGREES)
        max_lon = max(self._output_dataset.longitudes) + INPUT_MARGIN_DEGREES
        area=(max_lat, min_lon, min_lat, max_lon)
        
        self._input_dataset = open_dataset(input_anemoi_dataset_path, select=input_channel_names, start=start_date, end=end_date, area=area)
        assert self._input_dataset.shape[1] == len(input_channel_names)

        # Check that we have the same number of time points in each dataset
        assert self._input_dataset.shape[0] == self._output_dataset.shape[0]

        # Load static info and channel names
        if static_channel_names:
            self._static_channel_names = [ChannelMetadata(name) if len(name.split('_'))==1 
                                            else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                            for name in static_channel_names]
            static_dataset = open_dataset(target_anemoi_dataset_path, select=static_channel_names, start=start_date, end=start_date)
            assert static_dataset.shape[1] == len(static_channel_names)
            # take first time point, and squeeze() to remove ensemble dimension
            static_data = static_dataset[0,:,:,:].squeeze()
            # Could also get these from stats, but one-time calculation is OK.
            self.static_mean = static_data.mean(axis=-1, keepdims=True)
            self.static_std = static_data.std(axis=-1, keepdims=True)
            target_shape = static_dataset.field_shape
            self.static_data_normalized = (static_data - self.static_mean.reshape((self.static_mean.shape[0],1))) \
                                            / self.static_std.reshape((self.static_std.shape[0],1))
            
            # self.normalize_input(np.flip(static_data.squeeze().reshape(-1, *target_shape), 1))
        else:
            self.static_data_normalized = None
            self._static_channel_names = []

        # Load target channel names
        self._output_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                        else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                        for name in output_channel_names]
        # Load era5 channel names
        self._input_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                    else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                    for name in input_channel_names]
        
        # Load stats for normalizing channels of input and output
        cosmo_stats = self._output_dataset.statistics
        self.output_mean = cosmo_stats['mean'][:]
        self.output_std = cosmo_stats['stdev'][:]

        input_stats = self._input_dataset.statistics
        self.input_mean = input_stats['mean'][:]
        self.input_std = input_stats['stdev'][:]

        assert len(transform_channels) == len(transform_input_means) ==\
            len(transform_input_stdevs) == len(transform_output_means) == \
                len(transform_output_stdevs)

        # FEATURE: load the mean and std values for transformed channels and update the normalization statistics
        self.input_transforms = {}
        self.input_inverse_transforms = {}
        self.output_transforms = {}
        self.output_inverse_transforms = {}
        for transform_descriptor in transform_channels:
            channel, transformation = transform_descriptor.split('-')
            input_channel_idx = input_channel_names.index(channel) if channel in input_channel_names else None
            output_channel_idx = output_channel_names.index(channel) if channel in output_channel_names else None
            if transformation.startswith('box_cox'):
                lmbda_str = transformation.split('_')[-1]
                lmbda = float(transformation.split('_')[-1])/(10**(len(lmbda_str)-1))
                print(f"Applying Box-Cox transformation with lambda={lmbda} to channel {channel} (input idx: {input_channel_idx} ({input_channel_names[input_channel_idx]}), output idx: {output_channel_idx} ({output_channel_names[output_channel_idx]}))")
                if input_channel_idx is not None:
                    self.input_transforms[input_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_transform(x, lmbda)
                    self.input_inverse_transforms[input_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_inverse_transform(x, lmbda)
                    self.input_mean[input_channel_idx] = transform_input_means[transform_descriptor]
                    self.input_std[input_channel_idx] = transform_input_stdevs[transform_descriptor]
                if output_channel_idx is not None:
                    self.output_transforms[output_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_transform(x, lmbda)
                    self.output_inverse_transforms[output_channel_idx] = lambda x, lmbda=lmbda: self.box_cox_inverse_transform(x, lmbda)
                    self.output_mean[output_channel_idx] = transform_output_means[transform_descriptor]
                    self.output_std[output_channel_idx] = transform_output_stdevs[transform_descriptor]
            else:
                raise ValueError(f"Transformation: {transformation} for channel {channel} not implemented.")

        # Initialize the interpolator
        self.interpolator = GridData(
            self._input_dataset.longitudes,
            self._input_dataset.latitudes,
            self._output_dataset.longitudes,
            self._output_dataset.latitudes)


    # DO NOT SUBMIT: This is not implemented yet.
    # Question: Is it OK to change the signature to return 3 items?
    def __getitem__(self, idx):
        """Get input and target data. Transform and normalize, but do not interpolate."""

        # Pull input, replacing the corrected tp if applicable
        date_str = to_datetime(self._input_dataset.dates[idx]).strftime('%Y%m%d-%H%M')
        # Don't reshape, but do squeeze ensemble dimension.
        input_data = self._input_dataset[idx].squeeze()
        # TODO: Consider generalizing this to other channels, in case we have cp.
        if ChannelMetadata('tp') in self._input_channels:
            tp_idx = self._input_channels.index(ChannelMetadata('tp'))
            corrected_tp_data = np.load(os.path.join(self._corrected_tp_path, f'{date_str}.npy'))
            input_data[tp_idx,::] = corrected_tp_data
        input_data = self.normalize_input(input_data)
        
        # Pull target data
        # squeeze the ensemble dimesnsion
        # reshape to image_shape
        # flip so that it starts in top-left corner (by default it is bottom left)
        target_shape = self.image_shape()
        target_data = self._output_dataset[idx]
        target_data = np.flip(target_data \
                .squeeze() \
                .reshape(-1,*target_shape),
            1)
        target_data = self.normalize_output(target_data)

        return torch.from_numpy(target_data),\
                torch.from_numpy(input_data)
    
    def get_static_data(self):
        return self.static_data_normalized
    
    def __len__(self):
        return len(self._output_dataset.dates)

    # Question: Do we need an input longitude as well?
    def longitude(self) -> np.ndarray:
        """Get longitude values from the target dataset."""
        return self._output_dataset.longitudes

    def latitude(self) -> np.ndarray:
        """Get latitude values from the target dataset."""
        return self._output_dataset.latitudes

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return self._input_channels

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return self._output_channels

    def static_channels(self) -> List[ChannelMetadata]:
        """Metadata for the static channels. A list of ChannelMetadata, one for each channel"""
        return self._static_channel_names

    def time(self) -> List:
        """Get time values from the dataset."""
        #TODO Choose the time format and convert to that, currently it's a string from a filename
        return [to_datetime(dt64).strftime('%Y%m%d-%H%M') for dt64 in self._output_dataset.dates]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data."""
        return self._output_dataset.field_shape
    
    def input_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the input data."""
        return self._input_dataset.field_shape
    
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        for channel_idx, transform in self.input_transforms.items():
            x[channel_idx,::] = transform(x[channel_idx,::])
        return (x - self.input_mean[(...,) + (None,) * (x.ndim - 1)]) \
                / self.input_std[(...,) + (None,) * (x.ndim - 1)]


    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        x = x * self.input_std[(...,) + (None,) * (x.ndim - 1)] \
                + self.input_mean[(...,) + (None,) * (x.ndim - 1)]
        for channel_idx, inverse_transform in self.input_inverse_transforms.items():
            x[channel_idx,::] = inverse_transform(x[channel_idx,::])
        return x


    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        for channel_idx, transform in self.output_transforms.items():
            x[channel_idx,::] = transform(x[channel_idx,::])
        return (x - self.output_mean[(...,) + (None,) * (x.ndim - 1)]) \
                / self.output_std[(...,) + (None,) * (x.ndim - 1)]


    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        x = x * self.output_std[(...,) + (None,) * (x.ndim - 1)] \
                + self.output_mean[(...,) + (None,) * (x.ndim - 1)]
        for channel_idx, inverse_transform in self.output_inverse_transforms.items():
            x[channel_idx,::] = inverse_transform(x[channel_idx,::])
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

ANEMOI_ERA5_REAL = AnemoiDataset
ANEMOI_ERA5_COSMO = AnemoiDataset
