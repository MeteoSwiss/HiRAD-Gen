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

"""
Model architecture layers used in the paper "Elucidating the Design Space of 
Diffusion-Based Generative Models".
"""

import contextlib
import importlib
from typing import Any, Dict, List

import numpy as np
import nvtx
import torch
import torch.amp as amp
from einops import rearrange
from torch.nn.functional import elu, gelu, leaky_relu, relu, sigmoid, silu, tanh

from .utils import weight_init, _validate_amp


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_mode = amp_mode
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            _weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
        x = x @ weight.t()
        if self.bias is not None:
            x = x.add_(bias)
        return x


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of conv2d. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        fused_conv_bias: bool = False,
        amp_mode: bool = False,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        if not kernel and fused_conv_bias:
            print(
                "Warning: Kernel is required when fused_conv_bias is enabled. Setting fused_conv_bias to False."
            )
            fused_conv_bias = False

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.fused_conv_bias = fused_conv_bias
        self.amp_mode = amp_mode
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        weight, bias, resample_filter = self.weight, self.bias, self.resample_filter
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
            if (
                self.resample_filter is not None
                and self.resample_filter.dtype != x.dtype
            ):
                resample_filter = self.resample_filter.to(x.dtype)

        w = weight if weight is not None else None
        b = bias if bias is not None else None
        f = resample_filter if resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x, w, padding=max(w_pad - f_pad, 0), bias=b
                )
            else:
                x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making it suitable for batch sizes of any size
    or even for batch-free scenarios.

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
        Deprecated. Please use
        :func:`~physicsnemo.nn.get_group_norm` instead.
    fused_act : bool, optional, default=False
        Deprecated. Please use
        :func:`~physicsnemo.nn.get_group_norm` instead.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is
        enabled.

    Forward
    -------
    x : torch.Tensor
        4-D input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is ``num_channels``, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.

    .. note::

    If ``num_channels`` is not divisible by ``num_groups``, the actual number of
    groups might be adjusted to satisfy the ``min_channels_per_group`` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
        use_apex_gn: bool = False,
        fused_act: bool = False,
        act: str | None = None,
        amp_mode: bool = False,
    ):
        super().__init__()
        # backwards compatibility warnings
        if use_apex_gn:
            raise ValueError(
                "'use_apex_gn' is deprecated. Please use 'get_group_norm' to enable "
                "Apex-based group norm."
            )
        if fused_act:
            raise ValueError(
                "'fused_act' is deprecated and only supported for Apex-based group norm. "
                "Please use `get_group_norm` to enable fused activations."
            )

        # initialize groupnorm
        self.num_groups: int = _compute_groupnorm_groups(
            num_channels, num_groups, min_channels_per_group
        )
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.act = act.lower() if act else act
        self.act_fn = None
        if self.act is not None:
            self.act_fn = self.get_activation_function()
        self.amp_mode = amp_mode

    def forward(self, x):
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)

        if self.training:
            # Use default torch implementation of GroupNorm for training
            # This does not support channels last memory format
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=weight,
                bias=bias,
                eps=self.eps,
            )
        else:
            # Use custom GroupNorm implementation that supports channels last
            # memory layout for inference
            x = rearrange(x, "b (g c) h w -> b g c h w", g=self.num_groups)

            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)

            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c h w -> b (g c) h w")

            weight = rearrange(weight, "c -> 1 c 1 1")
            bias = rearrange(bias, "c -> 1 c 1 1")
            x = x * weight + bias

        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def get_activation_function(self):
        """
        Get activation function given string input
        """

        activation_map = {
            "silu": silu,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "gelu": gelu,
            "elu": elu,
        }

        act_fn = activation_map.get(self.act, None)
        if act_fn is None:
            raise ValueError(f"Unknown activation function: {self.act}")
        return act_fn


class Attention(torch.nn.Module):
    """
    Self-attention block used in U-Net-style architectures, such as DDPM++, NCSN++, and ADM.
    Applies GroupNorm followed by multi-head self-attention and a projection layer.

    Parameters
    ----------
    out_channels : int
        Number of channels :math:`C` in the input and output feature maps.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon value for numerical stability in GroupNorm.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.


    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is `out_channels`, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.
    """

    def __init__(
        self,
        *,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-5,
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        init: Dict[str, Any] = dict(),
        use_apex_gn: bool = False,
        amp_mode: bool = False,
        fused_conv_bias: bool = False,
    ) -> None:
        super().__init__()
        # Parameters validation
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, but got {num_heads}"
            )
        if out_channels % num_heads != 0:
            raise ValueError(
                f"`out_channels` must be divisible by `num_heads`, but got {out_channels} and {num_heads}"
            )
        self.num_heads = num_heads
        self.norm = get_group_norm(
            num_channels=out_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            amp_mode=amp_mode,
        )
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1: torch.Tensor = self.qkv(self.norm(x))

        q, k, v = (
            (
                x1.reshape(
                    x.shape[0], self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
            )
            .permute(0, 1, 4, 3, 2)
            .unbind(-2)
        )
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )
        attn = attn.transpose(-1, -2)

        x: torch.Tensor = self.proj(attn.reshape(*x.shape)).add_(x)
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    learnable : bool, optional
        A boolean flag indicating whether learnable positional embedding is enabled. Defaults to False.
    freq_embed_dim: int, optional
        The dimension of the frequency embedding. Defaults to None, in which case it will be set to num_channels.
    mlp_hidden_dim: int, optional
        The dimension of the hidden layer in the MLP. Defaults to None, in which case it will be set to 2 * num_channels.
        Only applicable if learnable is True; if learnable is False, this parameter is ignored.
    embed_fn: Literal["cos_sin", "np_sin_cos"], optional
        The function to use for embedding into sin/cos features (allows for swapping the order of sin/cos). Defaults to 'cos_sin'.
        Options:
            - 'cos_sin': Uses torch to compute frequency embeddings and returns in order (cos, sin)
            - 'np_sin_cos': Uses numpy to compute frequency embeddings and returns in order (sin, cos)
    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
        amp_mode: bool = False,
        learnable: bool = False,
        freq_embed_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        embed_fn: Literal["cos_sin", "np_sin_cos"] = "cos_sin",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.amp_mode = amp_mode
        self.learnable = learnable
        self.embed_fn = embed_fn

        if freq_embed_dim is None:
            freq_embed_dim = num_channels
        self.freq_embed_dim = freq_embed_dim

        if learnable:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = 2 * num_channels
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(freq_embed_dim, mlp_hidden_dim, bias=True),
                torch.nn.SiLU(),
                torch.nn.Linear(mlp_hidden_dim, num_channels, bias=True),
            )

        if self.embed_fn == "np_sin_cos":
            half_embed_dim = freq_embed_dim // 2
            pow = np.arange(half_embed_dim, dtype=np.float32) / half_embed_dim
            w = np.exp(-np.log(self.max_positions) * pow)
            self.register_buffer("freqs", torch.from_numpy(w).float())

    def _cos_sin_embedding(self, x):
        freqs = torch.arange(
            start=0, end=self.freq_embed_dim // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.freq_embed_dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if freqs.dtype != x.dtype:
                freqs = freqs.to(x.dtype)
        x = x.ger(freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

    def _sin_cos_embedding_np(self, x):
        x = torch.outer(x, self.freqs)
        x = torch.cat([x.sin(), x.cos()], dim=1)
        return x

    def forward(self, x):
        if self.embed_fn == "cos_sin":
            x = self._cos_sin_embedding(x)
        elif self.embed_fn == "np_sin_cos":
            x = self._sin_cos_embedding_np(x)

        if self.learnable:
            x = self.mlp(x)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(self, num_channels: int, scale: int = 16, amp_mode: bool = False):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)
        self.amp_mode = amp_mode

    def forward(self, x):
        freqs = self.freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if x.dtype != self.freqs.dtype:
                freqs = self.freqs.to(x.dtype)

        x = x.ger((2 * np.pi * freqs))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
