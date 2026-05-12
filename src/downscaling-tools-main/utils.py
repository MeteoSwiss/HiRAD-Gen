import torch
import numpy as np
import xarray as xr


def tensors_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            obj = obj.detach()
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, list):
        if all(isinstance(item, np.ndarray) for item in obj):
            return obj
        return [tensors_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(tensors_to_numpy(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: tensors_to_numpy(value) for key, value in obj.items()}
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
