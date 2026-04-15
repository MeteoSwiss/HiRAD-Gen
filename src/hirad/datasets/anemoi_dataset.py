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
from pathlib import Path
# import zarr

from .constants import REAL_TO_ERA_CHANNEL_MAP, ERA_TO_REAL_CHANNEL_MAP
from hirad.utils.console import PythonLogger
from hirad.utils.dataset_utils import GridData, regrid_icon_to_rotlatlon


logger = PythonLogger(__name__)

# Margin to use for ERA dataset (to avoid nans from interpolation at boundary)
INPUT_MARGIN_DEGREES = 0.5

class AnemoiDataset(DownscalingDataset):
    def __init__(self,
                type: str,
                input_anemoi_dataset_path: str,
                target_anemoi_dataset_path: str,
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

        input_dataset = type.split('_')[-2]
        target_dataset = type.split('_')[-1]
        self.real_target = target_dataset == 'real'
        self.trim_edge = trim_edge

        if input_dataset != 'era5':
            raise ValueError(f"Input dataset {input_dataset} not supported for AnemoiDataset. Only 'era5' is supported.")
        if target_dataset != 'cosmo' and target_dataset !='real':
            raise ValueError(f"Target dataset {target_dataset} not supported for AnemoiDataset. Only 'cosmo' and 'real' are supported.")

        if self.real_target:
            # Map output channel names from real to era5
            output_channel_names_real = [ERA_TO_REAL_CHANNEL_MAP[name] for name in output_channel_names]
            self.lat_lon_real = torch.load("/capstor/store/cscs/pasc/c38/real_grid_info/realch1-lat-lon", weights_only=False)
            self.regrid_indices_real = torch.from_numpy(np.load("/capstor/store/cscs/pasc/c38/real_grid_info/remap_indices.npy")).long()
            self.regrid_weights_real = torch.from_numpy(np.load("/capstor/store/cscs/pasc/c38/real_grid_info/remap_weights.npy"))

        #TODO switch hanbdling paths to Path rather than pure strings
        self._n_month_hour_channels = n_month_hour_channels
        target_open_dataset_kwargs = {}
        if start_date is not None and end_date is not None:
            assert start_date < end_date, "start_date must be before end_date"
            target_open_dataset_kwargs['start'] = start_date
            target_open_dataset_kwargs['end'] = end_date
        if trim_edge > 0 and not self.real_target:
            target_open_dataset_kwargs['trim_edge'] = trim_edge
        self._output_dataset = open_dataset(target_anemoi_dataset_path, select=output_channel_names_real if self.real_target else output_channel_names, **target_open_dataset_kwargs)
        assert self._output_dataset.shape[1] == len(output_channel_names)

        # Load ERA dataset, trimming the area and limiting the dates to the target dataset
        start_date = self._output_dataset.metadata()['start_date'] if start_date is None else start_date
        end_date = self._output_dataset.metadata()['end_date'] if end_date is None else end_date
        latitudes = self.latitude()
        longitudes = self.longitude()
        min_lat = min(latitudes) - INPUT_MARGIN_DEGREES
        max_lat = max(latitudes) + INPUT_MARGIN_DEGREES
        min_lon = max(0, min(longitudes) - INPUT_MARGIN_DEGREES)
        max_lon = max(longitudes) + INPUT_MARGIN_DEGREES
        area=(max_lat, min_lon, min_lat, max_lon)
        
        self._input_dataset = open_dataset(input_anemoi_dataset_path, select=input_channel_names, start=start_date, end=end_date, area=area)
        assert self._input_dataset.shape[1] == len(input_channel_names)

        # Check that we have the same number of time points in each dataset
        assert self._input_dataset.shape[0] == self._output_dataset.shape[0]

        # Load static info and channel names
        if static_channel_names:
            self._static_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                            else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                            for name in static_channel_names]
            static_open_dataset_kwargs = {}
            if not self.real_target and trim_edge > 0:
                static_open_dataset_kwargs['trim_edge'] = trim_edge
            static_dataset = open_dataset(target_anemoi_dataset_path, select=static_channel_names, start=start_date, end=start_date, **static_open_dataset_kwargs)
            assert static_dataset.shape[1] == len(static_channel_names)
            # take first time point, and squeeze() to remove ensemble dimension
            static_data = static_dataset[0,:,:,:].squeeze()
            # Could also get these from stats, but one-time calculation is OK.
            self.static_mean = static_data.mean(axis=-1, keepdims=True)
            self.static_std = static_data.std(axis=-1, keepdims=True)
            target_shape = self.image_shape()
            self.static_data_normalized = (static_data - self.static_mean.reshape((self.static_mean.shape[0],1))) \
                                            / self.static_std.reshape((self.static_std.shape[0],1))
            self.static_data_normalized = torch.from_numpy(self.static_data_normalized)
            self.static_data_normalized = regrid_icon_to_rotlatlon(self.static_data_normalized, self.regrid_indices_real, self.regrid_weights_real)
            if trim_edge > 0 and self.real_target:
                self.static_data_normalized = self.static_data_normalized[:, trim_edge:-trim_edge, trim_edge:-trim_edge]
            # self.normalize_input(np.flip(static_data.squeeze().reshape(-1, *target_shape), 1))
        else:
            self.static_data_normalized = None
            self._static_channels = []

        # Load target channel names
        self._output_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                        else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                        for name in output_channel_names]
        # Load era5 channel names
        self._input_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                    else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                    for name in input_channel_names]
        
        # Load stats for normalizing channels of input and output
        target_stats = self._output_dataset.statistics
        self.output_mean = target_stats['mean'][:]
        self.output_std = target_stats['stdev'][:]

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
            self.longitude(),
            self.latitude())


    # DO NOT SUBMIT: This is not implemented yet.
    # Question: Is it OK to change the signature to return 3 items?
    def __getitem__(self, idx):
        """Get input and target data. Transform and normalize, but do not interpolate."""

        # Pull input, replacing the corrected tp if applicable
        date_str = to_datetime(self._input_dataset.dates[idx]).strftime('%Y%m%d-%H%M')
        
        # Don't reshape, but do squeeze ensemble dimension.
        input_data = self._input_dataset[idx].squeeze()
        
        # Pull target data
        # squeeze the ensemble dimesnsion
        target_data = self._output_dataset[idx].squeeze()
        
        # next two steps only if target is cosmo, real has to be regridded first (done in training loop on gpu-s for efficiency)
        # reshape to image_shape
        # flip so that it starts in top-left corner (by default it is bottom left)
        if not self.real_target:
            target_shape = self.image_shape()
            target_data = np.flip(target_data \
                    .reshape(-1,*target_shape),
                1)

        return torch.from_numpy(target_data.copy()),\
                torch.from_numpy(input_data),\
                date_str
    
    def get_static_data(self):
        return self.static_data_normalized
    
    def __len__(self):
        return len(self._output_dataset.dates)

    # Question: Do we need an input longitude as well?
    def longitude(self) -> np.ndarray:
        """Get longitude values from the target dataset."""
        if self.real_target:
            return self.lat_lon_real[:,1]
        return self._output_dataset.longitudes

    def latitude(self) -> np.ndarray:
        """Get latitude values from the target dataset."""
        if self.real_target:
            return self.lat_lon_real[:,0]
        return self._output_dataset.latitudes

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return self._input_channels

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return self._output_channels

    def static_channels(self) -> List[ChannelMetadata]:
        """Metadata for the static channels. A list of ChannelMetadata, one for each channel"""
        return self._static_channels

    def time(self) -> List:
        """Get time values from the dataset."""
        #TODO Choose the time format and convert to that, currently it's a string from a filename
        return [to_datetime(dt64).strftime('%Y%m%d-%H%M') for dt64 in self._output_dataset.dates]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data."""
        if self.real_target:
            return 704,1088
        return self._output_dataset.field_shape
    
    def input_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the input data."""
        return self._input_dataset.field_shape

    def normalization_stats(self):
        """Get the mean and std stats for normalizing the input and output data."""
        return {"input_mean": self.input_mean,
                "input_std": self.input_std,
                "output_mean": self.output_mean,
                "output_std": self.output_std}

    def stats_to_torch(self, device: torch.device, dtype: torch.dtype = torch.float32):
        """Convert the mean and std stats to torch tensors on the specified device."""
        self.input_mean = torch.from_numpy(self.input_mean).to(device=device, dtype=dtype)
        self.input_std = torch.from_numpy(self.input_std).to(device=device, dtype=dtype)
        self.output_mean = torch.from_numpy(self.output_mean).to(device=device, dtype=dtype)
        self.output_std = torch.from_numpy(self.output_std).to(device=device, dtype=dtype)

    def stats_to_numpy(self):
        """Convert the mean and std stats to numpy arrays."""
        self.input_mean = self.input_mean.cpu().numpy() if isinstance(self.input_mean, torch.Tensor) else self.input_mean
        self.input_std = self.input_std.cpu().numpy() if isinstance(self.input_std, torch.Tensor) else self.input_std
        self.output_mean = self.output_mean.cpu().numpy() if isinstance(self.output_mean, torch.Tensor) else self.output_mean
        self.output_std = self.output_std.cpu().numpy() if isinstance(self.output_std, torch.Tensor) else self.output_std
    
    def normalize_input(self, x: np.ndarray | torch.Tensor, mean: np.ndarray | torch.Tensor = None, std: np.ndarray | torch.Tensor = None) -> np.ndarray | torch.Tensor:
        """Convert input from physical units to normalized data."""
        if mean is None:
            mean = self.input_mean
        if std is None:
            std = self.input_std
        for channel_idx, transform in self.input_transforms.items():
            x[:,channel_idx,::] = transform(x[:,channel_idx,::])
        return (x - self.input_mean[(None,) + (...,) + (None,) * (x.ndim - 2)]) \
                / self.input_std[(None,) + (...,) + (None,) * (x.ndim - 2)]


    def denormalize_input(self, x: np.ndarray | torch.Tensor, mean: np.ndarray | torch.Tensor = None, std: np.ndarray | torch.Tensor = None) -> np.ndarray | torch.Tensor:
        """Convert input from normalized data to physical units."""
        if mean is None:
            mean = self.input_mean
        if std is None:
            std = self.input_std
        x = x * self.input_std[(None,) + (...,) + (None,) * (x.ndim - 2)] \
                + self.input_mean[(None,) + (...,) + (None,) * (x.ndim - 2)]
        for channel_idx, inverse_transform in self.input_inverse_transforms.items():
            x[:,channel_idx,::] = inverse_transform(x[:,channel_idx,::])
        return x


    def normalize_output(self, x: np.ndarray | torch.Tensor, mean: np.ndarray | torch.Tensor = None, std: np.ndarray | torch.Tensor = None) -> np.ndarray | torch.Tensor:
        """Convert output from physical units to normalized data."""
        if mean is None:
            mean = self.output_mean
        if std is None:
            std = self.output_std
        for channel_idx, transform in self.output_transforms.items():
            x[:,channel_idx,::] = transform(x[:,channel_idx,::])
        return (x - self.output_mean[(None,) + (...,) + (None,) * (x.ndim - 2)]) \
                / self.output_std[(None,) + (...,) + (None,) * (x.ndim - 2)]


    def denormalize_output(self, x: np.ndarray | torch.Tensor, mean: np.ndarray | torch.Tensor = None, std: np.ndarray | torch.Tensor = None) -> np.ndarray | torch.Tensor:
        """Convert output from normalized data to physical units."""
        if mean is None:
            mean = self.output_mean
        if std is None:
            std = self.output_std
        x = x * self.output_std[(None,) + (...,) + (None,) * (x.ndim - 2)] \
                + self.output_mean[(None,) + (...,) + (None,) * (x.ndim - 2)]
        for channel_idx, inverse_transform in self.output_inverse_transforms.items():
            x[:,channel_idx,::] = inverse_transform(x[:,channel_idx,::])
        return x

    def box_cox_transform(self, channel_array: np.ndarray | torch.Tensor, lmbda: float) -> np.ndarray | torch.Tensor:
        """Apply Box-Cox transformation to the data."""
        if isinstance(channel_array, torch.Tensor):
            channel_array = torch.clamp(channel_array, min=0)
            return (torch.pow(channel_array, lmbda) - 1) / lmbda
        channel_array = np.clip(channel_array, 0, None)
        return (np.power(channel_array, lmbda) - 1) / lmbda

    def box_cox_inverse_transform(self, channel_array: np.ndarray | torch.Tensor, lmbda: float) -> np.ndarray | torch.Tensor:
        """Apply inverse Box-Cox transformation to the data."""
        if isinstance(channel_array, torch.Tensor):
            channel_array = torch.clamp(channel_array, min=-1/lmbda)
            return torch.pow((lmbda * channel_array) + 1, 1 / lmbda)
        channel_array = np.clip(channel_array, -1/lmbda, None)
        return np.power((lmbda * channel_array) + 1, 1 / lmbda)

    def make_time_grids(self, dates: list[str], device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Create multi-frequency cyclic sin/cos feature grids for hour and month (batched).

        Parameters
        ----------
        dates : Sequence[str]
            Date strings in the format 'YYYYMMDD-HHMM', length = B

        Returns
        -------
        grid : torch.Tensor, shape (B, C, H, W)
            Channels = [sin(k*hour), cos(k*hour), sin(k*month), cos(k*month) for each k]
        """

        B = len(dates)

        # --- parse month and hour ---
        months = torch.tensor(
            [int(d.split("-")[0][4:6]) for d in dates],
            dtype=torch.float32,
            device=device,
        )
        hours = torch.tensor(
            [int(d.split("-")[1][0:2]) for d in dates],
            dtype=torch.float32,
            device=device,
        )

        # normalize cyclic components
        hours = (hours % 24) / 24.0        # (B,)
        months = ((months - 1) % 12) / 12.0  # (B,)

        # frequencies
        n_freq = self._n_month_hour_channels // 2
        freqs = torch.arange(
            1, n_freq + 1, dtype=torch.float32, device=device
        )  # (K,)

        # shape helpers
        hours = hours[:, None]    # (B, 1)
        months = months[:, None]  # (B, 1)

        # --- hour encodings ---
        hour_angles = 2 * torch.pi * hours * freqs  # (B, K)
        hour_feats = torch.stack(
            [torch.sin(hour_angles), torch.cos(hour_angles)],
            dim=2
        )  # (B, K, 2)

        # --- month encodings ---
        month_angles = 2 * torch.pi * months * freqs
        month_feats = torch.stack(
            [torch.sin(month_angles), torch.cos(month_angles)],
            dim=2
        )  # (B, K, 2)

        # concatenate and flatten channels
        feats = torch.cat([hour_feats, month_feats], dim=1)  # (B, 2K, 2)
        feats = feats.reshape(B, -1)  # (B, C)

        # expand to spatial grid
        # grid = feats[:, :, None, None].expand(B, feats.shape[1], H, W)

        return feats

ANEMOI_ERA5_REAL = AnemoiDataset
ANEMOI_ERA5_COSMO = AnemoiDataset
