from .base import DownscalingDataset, ChannelMetadata
import os
import numpy as np
import torch
from typing import List, Tuple
import yaml

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
        print(cosmo_stats)
            

    def __len__(self):
        return len(self._file_list)


    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        lon_lat = torch.load(os.path.join(self._info_path,'cosmo-lat-lon'), weights_only=False)
        return lon_lat[:,0]


    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        lon_lat = torch.load(os.path.join(self._info_path,'cosmo-lat-lon'), weights_only=False)
        return lon_lat[:,1]


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
        #TODO load from info, I hardcode it for now
        return 390,582
    

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std


    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean


    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std


    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean