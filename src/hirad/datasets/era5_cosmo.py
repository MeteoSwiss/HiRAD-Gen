from .base import DownscalingDataset, ChannelMetadata
import os
import numpy as np
import torch
from typing import List, Tuple
import yaml
import torch.nn.functional as F

class ERA5_COSMO(DownscalingDataset):
    def __init__(self, dataset_path: str):
        super().__init__()

        #TODO switch hanbdling paths to Path rather than pure strings
        self._dataset_path = dataset_path
        self._era5_path = os.path.join(dataset_path, 'era-interpolated')
        self._cosmo_path = os.path.join(dataset_path, 'cosmo')
        self._info_path = os.path.join(dataset_path, 'info')

        # load file list (each file is one date-time state)
        self._file_list = os.listdir(self._cosmo_path)

        # Load cosmo info and channel names
        with open(os.path.join(self._info_path,'cosmo.yaml'), 'r') as file:
            self._cosmo_info = yaml.safe_load(file)
            self._cosmo_channels = [ChannelMetadata(name) for name in self._cosmo_info['select']]

        # Load era5 info and channel names
        with open(os.path.join(self._info_path,'era.yaml'), 'r') as file:
            self._era_info = yaml.safe_load(file)
            self._era_channels = [ChannelMetadata(name) if len(name.split('_'))==1 
                                 else ChannelMetadata(name.split('_')[0],name.split('_')[1])
                                   for name in self._era_info['select']]
        
        # Load stats for normalizing channels of input and output

        cosmo_stats = torch.load(os.path.join(self._info_path,'cosmo-stats'), weights_only=False)
        self.output_mean = cosmo_stats['mean']
        self.output_std = cosmo_stats['stdev']

        era_stats = torch.load(os.path.join(self._info_path,'era-stats'), weights_only=False)
        self.input_mean = era_stats['mean']
        self.input_std = era_stats['stdev']

    
    def __getitem__(self, idx):
        """Get cosmo and era5 interpolated to cosmo grid"""
        # get era5 data point
        # squeeze the ensemble dimesnsion
        # reshape to image_shape
        # flip so that it starts in top-left corner (by default it is bottom left)
        # orig_shape = [350,542] #TODO currently padding to be divisible by 16
        orig_shape = self.image_shape()
        era5_data = np.flip(torch.load(os.path.join(self._era5_path,self._file_list[idx]), weights_only=False)\
                                .squeeze() \
                                .reshape(-1,*orig_shape),
                            1)
        era5_data = self.normalize_input(era5_data)
        # get cosmo data point
        cosmo_data = np.flip(torch.load(os.path.join(self._cosmo_path,self._file_list[idx]), weights_only=False)\
                                .squeeze() \
                                .reshape(-1,*orig_shape),
                            1)
        cosmo_data = self.normalize_output(cosmo_data)
        # return samples
        return torch.tensor(cosmo_data),\
                torch.tensor(era5_data),\
                0
        # return F.pad(torch.tensor(cosmo_data), pad=(1,1,1,1), mode='constant', value=0), \
        #         F.pad(torch.tensor(era5_data), pad=(1,1,1,1), mode='constant', value=0), \
        #         0

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
        return self._era_channels


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
        return (x - self.input_mean.reshape((self.input_mean.shape[0],1,1))) \
                / self.input_std.reshape((self.input_std.shape[0],1,1))


    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std.reshape((self.input_std.shape[0],1,1)) \
                + self.input_mean.reshape((self.input_mean.shape[0],1,1))


    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean.reshape((self.output_mean.shape[0],1,1))) \
                / self.output_std.reshape((self.output_std.shape[0],1,1))


    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std.reshape((self.output_std.shape[0],1,1)) \
                + self.output_mean.reshape((self.output_mean.shape[0],1,1))