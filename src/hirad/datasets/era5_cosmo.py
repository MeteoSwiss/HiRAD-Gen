from hirad.datasets.base import DownscalingDataset, ChannelMetadata

class ERA5_COSMO(DownscalingDataset):
    def __init__(self):
        super().__init__()