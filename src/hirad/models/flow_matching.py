"""Flow-matching (rectified-flow) wrapper for super-resolution backbones.

This is the flow-matching counterpart of :class:`EDMPrecondSuperResolution`
(``preconditioning.py``). Where the EDM wrapper applies Karras-2022
preconditioning (c_skip/c_out/c_in/c_noise) and returns a *denoised* image, this
wrapper applies **no** preconditioning: the backbone directly predicts the
rectified-flow *velocity* field ``v_theta(x_t, t)`` for the linear probability
path

    x_t = (1 - t) * x_data + t * eps,   t in [0, 1],   eps ~ N(0, I)

so the training target is the (constant) velocity ``v = eps - x_data`` (see
:class:`hirad.losses.FlowMatchingLoss`).

Design notes
------------
* Construction mirrors ``EDMPrecondSuperResolution`` exactly (same backbone
  loaded by name, ``in_channels = img_in_channels + img_out_channels``), so a
  training manager can swap the wrapper class without touching channel bookkeeping.
* The conditioning image ``img_lr`` is concatenated to ``x_t`` *unscaled*
  (rectified flow feeds the raw noised state, no c_in). The result
  ``cat([x_t, img_lr])`` is the backbone input.
* The time label ``t`` is passed straight to the backbone's timestep embedder.
  The existing EDM-DiT feeds that same embedder O(1) inputs (``c_noise=log(sigma)/4``)
  and trains fine, and FM ``t in [0, 1]`` is the same magnitude regime, so no
  ADM-style ``t*1000`` rescale is required. ``time_scale`` (default 1.0) exposes
  this as a knob if a wider embedding range is ever wanted.
"""

import importlib
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn

network_module = importlib.import_module("hirad.models")


class FlowMatchingSuperResolution(nn.Module):
    """Rectified-flow velocity wrapper around a super-resolution backbone.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution ``(H, W)`` of the high-resolution image.
    img_in_channels : int
        Number of low-resolution conditioning channels (ERA5 + static [+ prev-HR]).
    img_out_channels : int
        Number of high-resolution output channels.
    use_fp16 : bool, optional
        Run the backbone in fp16 on CUDA, by default False.
    model_type : str, optional
        Backbone class name resolved from :mod:`hirad.models`, by default "DiT".
    time_scale : float, optional
        Multiplier applied to ``t`` before the backbone timestep embedder,
        by default 1.0 (pass ``t`` unchanged; see module docstring).

    Notes
    -----
    ``sigma_data``/``sigma_min``/``sigma_max`` are accepted and ignored so that
    a config/model_args dict shared with the EDM path round-trips cleanly. The
    flow-matching sampler does not use them; the harmless ``sigma_min=0.0`` /
    ``sigma_max=inf`` attributes are exposed only so generic sampler plumbing
    (which reads ``net.sigma_min``/``net.sigma_max``) does not crash.
    """

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal["DiT"] = "DiT",
        time_scale: float = 1.0,
        # accepted-and-ignored so an EDM-style model_args dict round-trips
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        **model_kwargs: dict,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.use_fp16 = use_fp16
        self.time_scale = time_scale
        # Exposed only for compatibility with generic sampler plumbing; unused by FM.
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        t: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """Predict the rectified-flow velocity ``v_theta(x_t, t)``.

        Parameters
        ----------
        x : torch.Tensor
            Noised high-resolution state ``x_t`` of shape (B, C_out, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning of shape (B, C_in, H, W). Concatenated to
            ``x`` unscaled.
        t : torch.Tensor
            Flow time in ``[0, 1]``; any shape broadcastable to (B,). Passed to the
            backbone timestep embedder as ``t * time_scale``.
        force_fp32 : bool, optional
            Force fp32 regardless of ``use_fp16``, by default False.
        **model_kwargs : dict
            Extra backbone kwargs, e.g. ``condition=<date embedding>``.

        Returns
        -------
        torch.Tensor
            Predicted velocity of shape (B, C_out, H, W).
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        if img_lr is None:
            arg = x
        else:
            arg = torch.cat([x, img_lr.to(x.dtype)], dim=1)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            t * self.time_scale,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        return F_x.to(torch.float32)

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """Passthrough for API parity with the EDM wrapper (FM uses no sigma grid)."""
        return torch.as_tensor(sigma)

    @property
    def amp_mode(self):
        """Return the *amp_mode* flag of the wrapped model or *None*."""
        return getattr(self.model, "amp_mode", None)
