# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import torch

from .layers import GroupNorm

_is_apex_available = False
if torch.cuda.is_available():
    try:
        apex_gn_module = importlib.import_module("apex.contrib.group_norm")
        ApexGroupNorm = getattr(apex_gn_module, "GroupNorm")
        _is_apex_available = True
    except ImportError:
        pass


def weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


def _recursive_property(prop_name: str, prop_type: type, doc: str) -> property:
    """
    Property factory that sets the property on a Module ``self`` and
    recursively on all submodules.
    For ``self``, the property is stored under a semi-private ``_<prop_name>`` attribute
    and for submodules the setter is delegated to the ``setattr`` function.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    prop_type : type
        The type of the property.
    doc : str
        The documentation string for the property.

    Returns
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        if not isinstance(value, prop_type):
            raise TypeError(
                f"{prop_name} must be a {prop_type.__name__} value, but got {type(value).__name__}."
            )
        # Set for self
        setattr(self, f"_{prop_name}", value)
        # Set for submodules
        submodules = iter(self.modules())
        next(submodules)  # Skip self
        for m in submodules:
            if hasattr(m, prop_name):
                setattr(m, prop_name, value)

    def _getter(self):
        return getattr(self, f"_{prop_name}")

    return property(_getter, _setter, doc=doc)


def _wrapped_property(prop_name: str, wrapped_obj_name: str, doc: str) -> property:
    """
    Property factory to define a property on a Module ``self`` that is
    wraps another Module in an attribute ``self.<wrapped_obj_name>``. The
    property delegates the setter and getter to the wrapped object's.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    wrapped_obj_name : str
        The name of the attribute that wraps the other Module.
    doc : str
        The documentation string for the property.

    Returns
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        wrapped_obj = getattr(self, wrapped_obj_name)
        if hasattr(wrapped_obj, prop_name):
            setattr(wrapped_obj, prop_name, value)
        else:
            raise AttributeError(f"{prop_name} is not supported by the wrapped model.")

    def _getter(self):
        wrapped_obj = getattr(self, wrapped_obj_name)
        return getattr(wrapped_obj, prop_name)

    return property(_getter, _setter, doc=doc)


def _validate_amp(amp_mode: bool) -> None:
    """Raise if `amp_mode` is False but PyTorch autocast (CPU or CUDA) is active.

    Parameters
    ----------
    amp_mode : bool
        Your intended AMP flag. Set False when you require full precision.
    """

    try:
        cuda_amp = bool(torch.is_autocast_enabled())
    except AttributeError:  # very old PyTorch
        cuda_amp = False
    try:
        cpu_amp = bool(torch.is_autocast_enabled("cpu"))
    except AttributeError:
        cpu_amp = False

    if not amp_mode and (cuda_amp or cpu_amp):
        active = []
        if cuda_amp:
            active.append("cuda")
        if cpu_amp:
            active.append("cpu")
        raise RuntimeError(
            f"amp_mode=False but torch autocast is enabled on: {', '.join(active)}. "
            "Disable autocast for this region or set amp_mode=True if mixed precision is intended."
        )


def get_group_norm(
    num_channels: int,
    num_groups: int = 32,
    min_channels_per_group: int = 4,
    eps: float = 1e-5,
    use_apex_gn: bool = False,
    act: str | None = None,
    amp_mode: bool = False,
) -> torch.nn.Module:
    """
    Utility function to get the GroupNorm layer, either from apex or from torch.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Desired number of groups to divide the input channels.
        This might be adjusted based on the ``min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number.
    eps : float, optional, default=1e-5
        A small number added to the variance to prevent division by zero.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.

    Returns
    -------
    torch.nn.Module
        The GroupNorm layer. If ``use_apex_gn`` is ``True``, returns an
        ApexGroupNorm layer, otherwise returns an instance of
        :class:`~physicsnemo.nn.GroupNorm`.

    .. note::

    If ``num_channels`` is not divisible by ``num_groups``, the actual number
    of groups might be adjusted to satisfy the ``min_channels_per_group``
    condition.
    """
    if use_apex_gn and not _is_apex_available:
        raise ValueError("'apex' is not installed, set `use_apex_gn=False`")

    act: str | None = act.lower() if act else act
    if use_apex_gn:
        # adjust number of groups to be consistent with GroupNorm
        num_groups: int = _compute_groupnorm_groups(
            num_channels, num_groups, min_channels_per_group
        )
        return ApexGroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=True,
            act=act,
        )
    else:
        return GroupNorm(
            num_channels=num_channels,
            num_groups=num_groups,
            min_channels_per_group=min_channels_per_group,
            eps=eps,
            act=act,
            amp_mode=amp_mode,
        )


def _compute_groupnorm_groups(
    num_channels: int,
    num_groups: int = 32,
    min_channels_per_group: int = 4,
) -> int:
    """
    Compute the number of groups for GroupNorm based on the number of channels
    and the minimum number of channels per group.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional, default=32
        Desired number of groups to divide the input channels.
        This might be adjusted based on the ``min_channels_per_group``.
    min_channels_per_group : int, optional, default=4
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number.

    Returns
    -------
    int
        The number of groups to use for GroupNorm.
    """
    num_groups: int = min(
        num_groups,
        (num_channels + min_channels_per_group - 1) // min_channels_per_group,
    )
    if num_channels % num_groups != 0:
        raise ValueError(
            "num_channels must be divisible by num_groups or min_channels_per_group"
        )
    return num_groups