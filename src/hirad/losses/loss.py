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


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from hirad.utils.patching import RandomPatching2D


class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self):
        """
        Arguments
        ----------
        """
        return

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        static_channels: Optional[torch.Tensor] = None,
        date_embedding: Optional[torch.Tensor] = None,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        use_apex_gn: bool = False,
    ) -> torch.Tensor:
        """
        Calculate and return the regression loss for
        deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        static_channels : torch.Tensor, optional
            Static channels input of shape (C_static, H, W).

        date_embedding : torch.Tensor, optional
            Date embedding input of shape (B, C_date).

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        Returns
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """
        weight = (
            1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)

        if static_channels is not None:
            y_lr = torch.cat(
                (y_lr, static_channels.expand(y_lr.shape[0], *static_channels.shape[1:])),
                dim=1,
            )

        if date_embedding is not None:
            date_embedding = date_embedding[:, :, None, None].expand(*date_embedding.shape[:2], *y_lr.shape[2:])
            if use_apex_gn:
                date_embedding = date_embedding.to(y_lr.dtype, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                date_embedding = date_embedding.to(y_lr.dtype, non_blocking=True).contiguous() 
            y_lr = torch.cat((y_lr, date_embedding), dim=1)

        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )
        
        loss = weight * ((D_yn - y) ** 2)

        return loss


class ResidualLoss:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching. It uses a pre-trained regression
    network to compute residuals before applying the diffusion process.

    Attributes
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise
            concatenated with `img_lr` for conditioning.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None

    def get_noise_params(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y) * sigma
        return n, sigma, weight

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        static_channels: Optional[torch.Tensor] = None,
        date_embedding: Optional[torch.Tensor] = None,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
        use_apex_gn: bool = False,
    ) -> torch.Tensor:
        """
        Calculate and return the loss for denoising score matching.

        This method computes a mixture loss that combines deterministic
        regression with denoising score matching. It first computes residuals
        using the regression network, then applies the diffusion process to
        these residuals.

        In addition to the standard denoising score matching loss, this method
        also supports optional patching for multi-diffusion. In this case, the spatial
        dimensions of the input are decomposed into `P` smaller patches of shape
        (H_patch, W_patch), that are grouped along the batch dimension, and the
        model is applied to each patch individually. In the following, if `patching`
        is not provided, then the input is not patched and `P=1` and `(H_patch,
        W_patch) = (H, W)`. When patching is used, the original non-patched conditioning is
        interpolated onto a spatial grid of shape `(H_patch, W_patch)` and channel-wise
        concatenated to the patched conditioning. This ensures that each patch
        maintains global information from the entire domain.

        The diffusion model `net` is expected to be conditioned on an input with
        `C_cond` channels, which should be:
            - `C_cond = C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is None.
            - `C_cond = C_hr + C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is None.
            - `C_cond = C_hr + 2*C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is not None.
            - `C_cond = 2*C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is not None.
        Additionally, `C_cond` should also include any embedding channels,
        such as positional embeddings or time embeddings.

        Note: this loss function does not apply any reduction.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Expected signature: `net(latent, y_lr, sigma,
            embedding_selector=embedding_selector, lead_time_label=lead_time_label,
            augment_labels=augment_labels)`, where:
                latent (torch.Tensor): Noisy input of shape (B[*P], C_hr, H_patch, W_patch)
                y_lr (torch.Tensor): Conditioning of shape (B[*P], C_cond, H_patch, W_patch)
                sigma (torch.Tensor): Noise level of shape (B[*P], 1, 1, 1)
                embedding_selector (callable, optional): Function to select
                    positional embeddings. Only used if `patching` is provided.
                lead_time_label (torch.Tensor, optional): Lead time labels.
                augment_labels (torch.Tensor, optional): Augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B[*P], C_hr, H_patch, W_patch)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the regression network and conditioning for the
            diffusion process.

        static_channels : Optional[torch.Tensor], optional
            Static channels input of shape (1, C_static, H, W), by default None.

        date_embedding : Optional[torch.Tensor], optional
            Date embedding input of shape (B, C_date), by default None

        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None. See
            :class:`physicsnemo.utils.patching.RandomPatching2D` for details.
            When provided, the patching strategy is used for both image patches
            and positional embeddings selection in the diffusion model `net`.
            Transforms tensors from shape (B, C, H, W) to (B*P, C, H_patch,
            W_patch).

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.

        augment_pipe : Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]
            Data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution images
                    of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels
        use_patch_grad_acc: bool, optional
            A boolean flag indicating whether to enable multi-iterations of patching accumulations
            for amortizing regression cost. Default False.
        use_apex_gn: bool, optional
            A boolean flag indicating whether apex group norm is used in the model.

        Returns
        -------
        torch.Tensor
            If patching is not used:
                A tensor of shape (B, C_hr, H, W) representing the per-sample loss.
            If patching is used:
                A tensor of shape (B*P, C_hr, H_patch, W_patch) representing
                the per-patch loss.

        Raises
        ------
        ValueError
            If patching is provided but is not an instance of RandomPatching2D.
            If shapes of img_clean and img_lr are incompatible.
        """

        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # print(f"Shape of y: {y.shape}, y_lr: {y_lr.shape}")

        # if using multi-iterations of patching, switch to optimized version
        if not use_patch_grad_acc or self.y_mean is None:
            # form residual
            if static_channels is not None:
                y_lr_res = torch.cat(
                    (y_lr_res, static_channels.expand(y_lr_res.shape[0], *static_channels.shape[1:])),
                    dim=1,
                )
            # print(f"Shape of y_lr after static channels regression: y_lr_res {y_lr_res.shape} y_lr {y_lr.shape}")

            if date_embedding is not None:
                date_embedding_reg = date_embedding[:, :, None, None].expand(*date_embedding.shape[:2], *y_lr_res.shape[2:])
                if use_apex_gn:
                    date_embedding_reg = date_embedding_reg.to(y_lr_res.dtype, non_blocking=True).to(memory_format=torch.channels_last)
                else:
                    date_embedding_reg = date_embedding_reg.to(y_lr_res.dtype, non_blocking=True).contiguous() 
                y_lr_res = torch.cat((y_lr_res, date_embedding_reg), dim=1)

            # print(f"Shape of y_lr after date embedding regression: y_lr_res {y_lr_res.shape} y_lr {y_lr.shape}")
            
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )

            self.y_mean = y_mean

        y = y - self.y_mean

        # print(f"Shape of y after residual: y {y.shape} y_lr {y_lr.shape}")

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # print(f"Shape of y_lr after hr mean conditioning: y_lr {y_lr.shape}")

        if static_channels is not None:
            y_lr = torch.cat(
                (y_lr, static_channels.expand(y_lr.shape[0], *static_channels.shape[1:])),
                dim=1,
            )

        # print(f"Shape of y_lr after static channels diffusion: y_lr {y_lr.shape}")

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        # removed patch_embedding_selector due to compilation issue with dynamo.
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            if static_channels is not None:
                img_lr = torch.cat(
                    (img_lr, static_channels.expand(img_lr.shape[0], *static_channels.shape[1:])),
                    dim=1,
                )
            # print(f"Shape of img_lr after static channels diffusion patching: img_lr {img_lr.shape}")
            if date_embedding is not None:
                date_embedding = date_embedding[:, :, None, None].expand(*date_embedding.shape[:2], *img_lr.shape[2:])
                if use_apex_gn:
                    date_embedding = date_embedding.to(img_lr.dtype, non_blocking=True).to(memory_format=torch.channels_last)
                else:
                    date_embedding = date_embedding.to(img_lr.dtype, non_blocking=True).contiguous() 
                img_lr = torch.cat((img_lr, date_embedding), dim=1)
            # print(f"Shape of img_lr after date embedding diffusion patching: img_lr {img_lr.shape}")
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)

            y = y_patched
            y_lr = y_lr_patched

        elif date_embedding is not None:
            date_embedding = date_embedding[:, :, None, None].expand(*date_embedding.shape[:2], *y_lr.shape[2:])
            if use_apex_gn:
                date_embedding = date_embedding.to(y_lr.dtype, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                date_embedding = date_embedding.to(y_lr.dtype, non_blocking=True).contiguous() 
            y_lr = torch.cat((y_lr, date_embedding), dim=1)

        # print(f"Final shapes before noise addition: y {y.shape} y_lr {y_lr.shape}")

        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)

        if lead_time_label is not None:
            D_yn = net(
                y + n,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                y + n,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                augment_labels=augment_labels,
            )
        loss = weight * ((D_yn - y) ** 2)

        return loss


class DiffusionLoss:
    """
    Diffusion loss function for training diffusion models.

    This class implements the standard loss function used for training
    diffusion models, which is based on denoising score matching. It computes
    the loss by adding noise to the input and comparing the model's predictions
    to the original clean input.

    Attributes
    ----------
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        channel_weights: Optional[list] = None,
    ):
        """
        Arguments
        ----------
        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        channel_weights : Optional[list], optional
            Per-output-channel multipliers applied to the loss (e.g. [1, 1, 1, 3]
            to upweight precipitation). None (default) applies uniform weighting.
        """
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.channel_weights = channel_weights
        self._channel_weights_t = None

    def apply_channel_weights(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply the per-pixel loss (B, C, H, W) by the per-channel weights."""
        if self.channel_weights is None:
            return loss
        if (
            self._channel_weights_t is None
            or self._channel_weights_t.device != loss.device
        ):
            # list() handles plain lists and OmegaConf ListConfig alike
            self._channel_weights_t = torch.as_tensor(
                list(self.channel_weights), device=loss.device, dtype=loss.dtype
            ).view(1, -1, 1, 1)
        return loss * self._channel_weights_t

    def get_noise_params(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y) * sigma
        return n, sigma, weight


    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        static_channels: Optional[torch.Tensor] = None,
        date_embedding: Optional[torch.Tensor] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculate and return the loss for denoising score matching.

        Note: this loss function does not apply any reduction.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Expected signature: `net(latent, y_lr, sigma, condition)`, where:
                latent (torch.Tensor): Noisy input of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Conditioning of shape (B, C_lr, H, W)
                sigma (torch.Tensor): Noise level of shape (B, 1, 1, 1)
                condition(torch.Tensor): Additional conditioning information, such as date-time embeddings,
                                         lead time embeddings (B, C_cond).
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W).
        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used for ground truth.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the regression network and conditioning for the
            diffusion process.

        static_channels : Optional[torch.Tensor], optional
            Static channels input of shape (1, C_static, H, W), by default None.

        date_embedding : Optional[torch.Tensor], optional
            Date embedding input of shape (B, C_date), by default None

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.


        Returns
        -------
        torch.Tensor
            A tensor of shape (B, C_hr, H, W) representing the per-sample loss.

        Raises
        ------
        ValueError
            If shapes of img_clean and img_lr are incompatible.
        """

        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        y = img_clean
        y_lr = img_lr

        if static_channels is not None:
            y_lr = torch.cat(
                (y_lr, static_channels.expand(y_lr.shape[0], *static_channels.shape[1:])),
                dim=1,
            )

        # create condition vector from date embedding and lead time label
        condition = None
        if date_embedding is not None:
            condition = date_embedding
        if lead_time_label is not None:
            if condition is not None:
                condition = torch.cat((condition, lead_time_label), dim=1)
            else:
                condition = lead_time_label

        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)

        D_yn = net(
            y + n,
            y_lr,
            sigma,
            condition=condition,
        )

        loss = weight * ((D_yn - y) ** 2)

        return self.apply_channel_weights(loss)


class AnchoredDiffusionLoss(DiffusionLoss):
    """
    Anchored (residual) EDM loss for the DiT: CorrDiff's two-stage mechanism on the
    diffusion-transformer backbone.

    A frozen pre-trained regression network provides a deterministic mean; the
    diffusion target is the per-channel STANDARDIZED residual

        y = (img_clean - y_mean) / residual_stds

    so every channel is ~unit variance and scalar ``sigma_data=1.0`` is correct.
    The regression mean is channel-wise concatenated to the conditioning
    (`hr_mean_conditioning`, as in :class:`ResidualLoss`), while the date embedding
    stays a global AdaLN `condition` vector (as in :class:`DiffusionLoss`).

    At inference the prediction is reconstructed as
    ``regression_mean + residual_stds * D_x`` before denormalization.

    The regression input is built exactly as in :class:`ResidualLoss` /
    :class:`RegressionLoss`: cat(img_lr, static_channels, date broadcast to HxW).
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        residual_stds: list,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 1.0,
        hr_mean_conditioning: bool = True,
        channel_weights: Optional[list] = None,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Frozen pre-trained regression network (eval, requires_grad=False).
            Same call signature as in :class:`ResidualLoss`.

        residual_stds : list
            Per-output-channel std of (normalized target - regression output),
            measured over a season-spanning training-period sample. The residual is
            divided by these so the diffusion target is unit variance per channel.

        P_mean, P_std : float, optional
            EDM noise-level distribution; defaults follow :class:`ResidualLoss`
            (P_mean=0.0), which trains harder at high sigma than the plain
            DiffusionLoss default (-1.2).

        sigma_data : float, optional
            1.0 by default -- correct for the standardized residual. Must match the
            EDMPrecondSuperResolution preconditioner.

        hr_mean_conditioning : bool, optional
            Concatenate the regression mean to the conditioning channels.

        channel_weights : Optional[list], optional
            Per-channel loss multipliers (see :class:`DiffusionLoss`).
        """
        super().__init__(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data,
                         channel_weights=channel_weights)
        self.regression_net = regression_net
        self.residual_stds = residual_stds
        self._residual_stds_t = None
        self.hr_mean_conditioning = hr_mean_conditioning
        # Compatibility with the validation loop, which resets loss_fn.y_mean.
        self.y_mean = None

    def _stds(self, ref: torch.Tensor) -> torch.Tensor:
        if self._residual_stds_t is None or self._residual_stds_t.device != ref.device:
            self._residual_stds_t = torch.as_tensor(
                list(self.residual_stds), device=ref.device, dtype=torch.float32
            ).view(1, -1, 1, 1)
        return self._residual_stds_t.to(ref.dtype)

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        static_channels: Optional[torch.Tensor] = None,
        date_embedding: Optional[torch.Tensor] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        use_apex_gn: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # ---- Frozen regression mean (input built exactly as in ResidualLoss) ----
        y_lr_res = img_lr
        if static_channels is not None:
            y_lr_res = torch.cat(
                (y_lr_res, static_channels.expand(y_lr_res.shape[0], *static_channels.shape[1:])),
                dim=1,
            )
        if date_embedding is not None:
            date_embedding_reg = date_embedding[:, :, None, None].expand(
                *date_embedding.shape[:2], *y_lr_res.shape[2:]
            )
            if use_apex_gn:
                date_embedding_reg = date_embedding_reg.to(
                    y_lr_res.dtype, non_blocking=True
                ).to(memory_format=torch.channels_last)
            else:
                date_embedding_reg = date_embedding_reg.to(
                    y_lr_res.dtype, non_blocking=True
                ).contiguous()
            y_lr_res = torch.cat((y_lr_res, date_embedding_reg), dim=1)

        with torch.no_grad():
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(img_clean), y_lr_res,
                    lead_time_label=lead_time_label,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(img_clean), y_lr_res,
                )
        self.y_mean = y_mean

        # ---- Standardized residual target ----
        y = (img_clean - y_mean) / self._stds(img_clean)

        # ---- Conditioning: [y_mean, img_lr, static]; date stays a condition vector ----
        y_lr = torch.cat((y_mean, img_lr), dim=1) if self.hr_mean_conditioning else img_lr
        if static_channels is not None:
            y_lr = torch.cat(
                (y_lr, static_channels.expand(y_lr.shape[0], *static_channels.shape[1:])),
                dim=1,
            )

        condition = None
        if date_embedding is not None:
            condition = date_embedding
        if lead_time_label is not None:
            condition = (
                torch.cat((condition, lead_time_label), dim=1)
                if condition is not None else lead_time_label
            )

        n, sigma, weight = self.get_noise_params(y)

        D_yn = net(
            y + n,
            y_lr,
            sigma,
            condition=condition,
        )

        loss = weight * ((D_yn - y) ** 2)

        return self.apply_channel_weights(loss)


class FlowMatchingLoss:
    """Rectified-flow (linear-path conditional flow matching) loss for the DiT.

    The flow-matching counterpart of :class:`DiffusionLoss`. Instead of EDM
    denoising score matching, it trains the network to predict the constant
    velocity of the linear probability path connecting data (``t=0``) and
    Gaussian noise (``t=1``):

        x_t = (1 - t) * img_clean + t * eps,    eps ~ N(0, I)
        v_target = d x_t / d t = eps - img_clean

    The network (:class:`~hirad.models.FlowMatchingSuperResolution` wrapping a
    DiT) predicts ``v_theta(x_t, t)`` and the loss is a plain (unweighted) MSE
    ``||v_theta - v_target||^2``.

    Conditioning mirrors :class:`DiffusionLoss` exactly: ``y_lr =
    cat(img_lr, static_channels)`` is the spatial conditioning that gets
    channel-concatenated to ``x_t`` inside the wrapper, while the date / lead-time
    embedding is passed as a global AdaLN ``condition`` vector.

    Timestep sampling follows Stable Diffusion 3 (Esser et al. 2024): ``t`` is
    drawn from a logit-normal distribution ``t = sigmoid(m + s * N(0,1))`` which
    concentrates samples near the middle of the path. Set
    ``time_sampling="uniform"`` for the plain rectified-flow schedule.

    Attributes
    ----------
    time_sampling : str
        ``"logit_normal"`` (default) or ``"uniform"``.
    logit_m, logit_s : float
        Location/scale of the logit-normal timestep distribution (SD3 defaults
        ``m=0.0``, ``s=1.0``). Ignored when ``time_sampling="uniform"``.
    channel_weights : Optional[list]
        Per-output-channel multipliers applied to the loss, e.g. ``[1, 1, 1, 3]``
        to upweight precipitation. ``None`` applies uniform weighting.
    """

    def __init__(
        self,
        time_sampling: str = "logit_normal",
        logit_m: float = 0.0,
        logit_s: float = 1.0,
        channel_weights: Optional[list] = None,
    ):
        if time_sampling not in ("logit_normal", "uniform"):
            raise ValueError(
                f"time_sampling must be 'logit_normal' or 'uniform', got {time_sampling!r}"
            )
        self.time_sampling = time_sampling
        self.logit_m = logit_m
        self.logit_s = logit_s
        self.channel_weights = channel_weights
        self._channel_weights_t = None

    def apply_channel_weights(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply the per-pixel loss (B, C, H, W) by the per-channel weights."""
        if self.channel_weights is None:
            return loss
        if (
            self._channel_weights_t is None
            or self._channel_weights_t.device != loss.device
        ):
            self._channel_weights_t = torch.as_tensor(
                list(self.channel_weights), device=loss.device, dtype=loss.dtype
            ).view(1, -1, 1, 1)
        return loss * self._channel_weights_t

    def sample_time(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Draw flow times ``t`` of shape (B, 1, 1, 1) in (0, 1)."""
        if self.time_sampling == "uniform":
            return torch.rand([batch_size, 1, 1, 1], device=device, dtype=dtype)
        # logit-normal (SD3)
        rnd = torch.randn([batch_size, 1, 1, 1], device=device, dtype=dtype)
        return torch.sigmoid(self.logit_m + self.logit_s * rnd)

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        static_channels: Optional[torch.Tensor] = None,
        date_embedding: Optional[torch.Tensor] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the per-pixel rectified-flow velocity-matching loss.

        Parameters
        ----------
        net : torch.nn.Module
            Flow-matching network. Expected signature
            ``net(x_t, y_lr, t, condition=...)`` returning the predicted velocity
            of shape (B, C_hr, H, W).
        img_clean : torch.Tensor
            High-resolution target of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning of shape (B, C_lr, H, W).
        static_channels : Optional[torch.Tensor]
            Static channels of shape (1, C_static, H, W), by default None.
        date_embedding : Optional[torch.Tensor]
            Date embedding of shape (B, C_date), by default None.
        lead_time_label : Optional[torch.Tensor]
            Lead-time embedding, by default None.

        Returns
        -------
        torch.Tensor
            Per-pixel loss of shape (B, C_hr, H, W) (no reduction).
        """
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. Batch size, height and width must match."
            )

        y = img_clean
        y_lr = img_lr
        if static_channels is not None:
            y_lr = torch.cat(
                (y_lr, static_channels.expand(y_lr.shape[0], *static_channels.shape[1:])),
                dim=1,
            )

        # Global AdaLN condition vector (date + optional lead-time)
        condition = None
        if date_embedding is not None:
            condition = date_embedding
        if lead_time_label is not None:
            condition = (
                torch.cat((condition, lead_time_label), dim=1)
                if condition is not None else lead_time_label
            )

        # Linear probability path: t=0 -> data, t=1 -> noise
        t = self.sample_time(y.shape[0], y.device, y.dtype)
        eps = torch.randn_like(y)
        x_t = (1.0 - t) * y + t * eps
        v_target = eps - y

        v_pred = net(x_t, y_lr, t, condition=condition)

        loss = (v_pred - v_target) ** 2
        return self.apply_channel_weights(loss)