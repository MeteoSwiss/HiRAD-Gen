# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

from hirad.losses.loss import RegressionLoss, ResidualLoss
from hirad.utils.patching import RandomPatching2D


# ---------------------------------------------------------------------------
#  Helpers / fixtures
# ---------------------------------------------------------------------------

B, C_HR, C_LR, H, W = 2, 3, 4, 64, 64


def _make_dummy_net(out_channels=C_HR):
    """Return a callable that mimics a neural network."""
    net = MagicMock()
    net.side_effect = lambda x, y_lr, *a, **kw: torch.zeros(
        x.shape[0], out_channels, x.shape[2], x.shape[3], device=x.device
    )
    return net


def _make_identity_augment():
    """Augmentation pipe that returns its input unchanged."""
    return lambda x: (x, None)


@pytest.fixture()
def img_clean():
    return torch.randn(B, C_HR, H, W)


@pytest.fixture()
def img_lr():
    return torch.randn(B, C_LR, H, W)


@pytest.fixture()
def static_channels():
    return torch.randn(1, 2, H, W)


@pytest.fixture()
def date_embedding():
    return torch.randn(B, 5)


@pytest.fixture()
def lead_time_label():
    return torch.randint(49, size=(B,))


############################################################################
#                          RegressionLoss                                  #
############################################################################


class TestRegressionLossBasic:
    """Basic behaviour of RegressionLoss."""

    def test_output_shape(self, img_clean, img_lr):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result = loss(net, img_clean, img_lr)
        assert result.shape == img_clean.shape

    def test_loss_non_negative(self, img_clean, img_lr):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result = loss(net, img_clean, img_lr)
        assert (result >= 0).all()

    def test_zero_loss_when_prediction_matches(self, img_clean, img_lr):
        """If net returns img_clean exactly, the loss should be zero."""
        net = MagicMock()
        # Net always returns the ground truth
        net.side_effect = lambda x, y_lr, *a, **kw: img_clean
        loss = RegressionLoss()
        result = loss(net, img_clean, img_lr)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_net_receives_zero_input(self, img_clean, img_lr):
        """First argument to the network should be a zero tensor."""
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr)
        call_args = net.call_args
        zero_input = call_args[0][0]
        torch.testing.assert_close(zero_input, torch.zeros_like(zero_input))

    def test_net_receives_lr_conditioning(self, img_clean, img_lr):
        """Second argument to the network should contain the LR image."""
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr)
        call_args = net.call_args
        y_lr_arg = call_args[0][1]
        # Without static channels or date embedding, the conditioning
        # should match the LR image.
        torch.testing.assert_close(y_lr_arg, img_lr)


class TestRegressionLossAugmentation:
    """Augmentation behaviour of RegressionLoss."""

    def test_identity_augmentation_same_as_no_augmentation(self, img_clean, img_lr):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result_a = loss(net, img_clean, img_lr, augment_pipe=None)
        result_b = loss(net, img_clean, img_lr, augment_pipe=_make_identity_augment())
        torch.testing.assert_close(result_a, result_b)

    def test_augment_pipe_is_called(self, img_clean, img_lr):
        augment_pipe = MagicMock(side_effect=_make_identity_augment())
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, augment_pipe=augment_pipe)
        augment_pipe.assert_called_once()


class TestRegressionLossStaticChannels:
    """Static-channel conditioning for RegressionLoss."""

    def test_lr_conditioning_includes_static_channels(
        self, img_clean, img_lr, static_channels
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, static_channels=static_channels)
        y_lr_arg = net.call_args[0][1]
        torch.testing.assert_close(
            y_lr_arg, torch.cat([img_lr, static_channels.expand(img_lr.shape[0], -1, -1, -1)], dim=1))

    def test_output_shape_with_static_channels(
        self, img_clean, img_lr, static_channels
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result = loss(net, img_clean, img_lr, static_channels=static_channels)
        assert result.shape == img_clean.shape


class TestRegressionLossDateEmbedding:
    """Date-embedding conditioning for RegressionLoss."""

    def test_lr_conditioning_includes_date_embedding(
        self, img_clean, img_lr, date_embedding
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, date_embedding=date_embedding)
        y_lr_arg = net.call_args[0][1]
        torch.testing.assert_close(
            y_lr_arg, torch.cat([img_lr, date_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)], dim=1)
        )

    def test_output_shape_with_date_embedding(
        self, img_clean, img_lr, date_embedding
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result = loss(net, img_clean, img_lr, date_embedding=date_embedding)
        assert result.shape == img_clean.shape

    def test_date_embedding_contiguous_without_apex(
        self, img_clean, img_lr, date_embedding
    ):
        """Date embedding should be contiguous when not using apex GN."""
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, date_embedding=date_embedding, use_apex_gn=False)
        y_lr_arg = net.call_args[0][1]
        assert y_lr_arg.is_contiguous()


class TestRegressionLossLeadTime:
    """Lead-time-label behaviour in RegressionLoss."""

    def test_lead_time_label_passed_to_net(
        self, img_clean, img_lr, lead_time_label
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, lead_time_label=lead_time_label)
        kw = net.call_args[1]
        assert "lead_time_label" in kw
        torch.testing.assert_close(kw["lead_time_label"], lead_time_label)

    def test_no_lead_time_label_omitted_from_kwargs(self, img_clean, img_lr):
        net = _make_dummy_net()
        loss = RegressionLoss()
        loss(net, img_clean, img_lr, lead_time_label=None)
        kw = net.call_args[1]
        assert "lead_time_label" not in kw


class TestRegressionLossCombined:
    """Combined optional arguments for RegressionLoss."""

    def test_all_optional_args(
        self, img_clean, img_lr, static_channels, date_embedding, lead_time_label
    ):
        net = _make_dummy_net()
        loss = RegressionLoss()
        result = loss(
            net,
            img_clean,
            img_lr,
            static_channels=static_channels,
            date_embedding=date_embedding,
            lead_time_label=lead_time_label,
        )
        assert result.shape == img_clean.shape
        y_lr_arg = net.call_args[0][1]
        expected_channels = C_LR + static_channels.shape[1] + date_embedding.shape[1]
        assert y_lr_arg.shape[1] == expected_channels


############################################################################
#                       ResidualLoss — init                                #
############################################################################


class TestResidualLossInit:
    """Tests for ResidualLoss initialization."""

    def test_default_values(self):
        reg_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        assert loss.P_mean == 0.0
        assert loss.P_std == 1.2
        assert loss.sigma_data == 0.5
        assert loss.hr_mean_conditioning is False
        assert loss.y_mean is None

    def test_custom_values(self):
        reg_net = _make_dummy_net()
        loss = ResidualLoss(
            regression_net=reg_net,
            P_mean=1.0,
            P_std=2.0,
            sigma_data=1.0,
            hr_mean_conditioning=True,
        )
        assert loss.P_mean == 1.0
        assert loss.P_std == 2.0
        assert loss.sigma_data == 1.0
        assert loss.hr_mean_conditioning is True


############################################################################
#                   ResidualLoss — get_noise_params                        #
############################################################################


class TestResidualLossGetNoiseParams:
    """Tests for ResidualLoss.get_noise_params."""

    def test_return_shapes(self):
        reg_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        y = torch.randn(B, C_HR, H, W)
        n, sigma, weight = loss.get_noise_params(y)
        assert n.shape == y.shape
        assert sigma.shape == (B, 1, 1, 1)
        assert weight.shape == (B, 1, 1, 1)

    def test_weight_and_sigma_positive(self):
        reg_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        y = torch.randn(B, C_HR, H, W)
        _, sigma, weight = loss.get_noise_params(y)
        assert (weight > 0).all()
        assert (sigma > 0).all()


############################################################################
#                    ResidualLoss — __call__ basic                         #
############################################################################


class TestResidualLossCallBasic:
    """Basic behaviour of ResidualLoss.__call__."""

    def test_output_shape_no_patching(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(diff_net, img_clean, img_lr)
        assert result.shape == img_clean.shape

    def test_loss_non_negative(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(diff_net, img_clean, img_lr)
        assert (result >= 0).all()

    def test_regression_net_called(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr)
        reg_net.assert_called_once()

    def test_diffusion_net_called(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr)
        diff_net.assert_called_once()

    def test_regression_net_receives_zero_input(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr)
        zero_input = reg_net.call_args[0][0]
        torch.testing.assert_close(zero_input, torch.zeros_like(zero_input))


############################################################################
#                  ResidualLoss — shape validation                         #
############################################################################


class TestResidualLossShapeValidation:
    """Validation of img_clean / img_lr shapes."""

    def test_batch_mismatch_raises(self):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        img_clean = torch.randn(2, C_HR, H, W)
        img_lr = torch.randn(3, C_LR, H, W)
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss(diff_net, img_clean, img_lr)

    def test_spatial_mismatch_raises(self):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        img_clean = torch.randn(B, C_HR, 32, 32)
        img_lr = torch.randn(B, C_LR, 64, 64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss(diff_net, img_clean, img_lr)

    def test_invalid_patching_type_raises(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        with pytest.raises(ValueError, match="RandomPatching2D"):
            loss(diff_net, img_clean, img_lr, patching="not_a_patching_object")


############################################################################
#                  ResidualLoss — augmentation                             #
############################################################################


class TestResidualLossAugmentation:
    """Augmentation in ResidualLoss."""

    def test_augment_pipe_called(self, img_clean, img_lr):
        augment_pipe = MagicMock(side_effect=_make_identity_augment())
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, augment_pipe=augment_pipe)
        augment_pipe.assert_called_once()

    def test_identity_augmentation_output_shape(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(
            diff_net,
            img_clean,
            img_lr,
            augment_pipe=_make_identity_augment(),
        )
        assert result.shape == img_clean.shape

    def test_identity_augmentation_same_as_no_augmentation(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        torch.manual_seed(0) # fix the seed because of random noise in the loss
        result_a = loss(diff_net, img_clean, img_lr, augment_pipe=None)
        torch.manual_seed(0) # reset the seed to ensure same noise is added
        result_b = loss(diff_net, img_clean, img_lr, augment_pipe=_make_identity_augment())
        torch.testing.assert_close(result_a, result_b)


############################################################################
#               ResidualLoss — static channels                             #
############################################################################


class TestResidualLossStaticChannels:
    """Static-channel conditioning in ResidualLoss."""

    def test_output_shape_with_static_channels(
        self, img_clean, img_lr, static_channels
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(diff_net, img_clean, img_lr, static_channels=static_channels)
        assert result.shape == img_clean.shape

    def test_regression_net_receives_static_channels(
        self, img_clean, img_lr, static_channels
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, static_channels=static_channels)
        y_lr_reg = reg_net.call_args[0][1]
        assert y_lr_reg.shape[1] == C_LR + static_channels.shape[1]
        torch.testing.assert_close(
            y_lr_reg[0, C_LR:, :, :], static_channels[0, :, :, :]
        )
        torch.testing.assert_close(
            y_lr_reg, torch.cat([img_lr, static_channels.expand(img_lr.shape[0], -1, -1, -1)], dim=1)
        )

    def test_diffusion_net_receives_static_channels(
        self, img_clean, img_lr, static_channels
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, static_channels=static_channels)
        y_lr_diff = diff_net.call_args[0][1]
        # Diffusion net also gets static channels appended to y_lr
        assert y_lr_diff.shape[1] == C_LR + static_channels.shape[1]
        torch.testing.assert_close(
            y_lr_diff[0, C_LR:, :, :], static_channels[0, :, :, :]
        )
        torch.testing.assert_close(
            y_lr_diff, torch.cat([img_lr, static_channels.expand(img_lr.shape[0], -1, -1, -1)], dim=1)
        )

############################################################################
#                ResidualLoss — date embedding                             #
############################################################################


class TestResidualLossDateEmbedding:
    """Date-embedding conditioning in ResidualLoss."""

    def test_output_shape_with_date_embedding(
        self, img_clean, img_lr, date_embedding
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(diff_net, img_clean, img_lr, date_embedding=date_embedding)
        assert result.shape == img_clean.shape

    def test_regression_net_receives_date_embedding(
        self, img_clean, img_lr, date_embedding
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, date_embedding=date_embedding)
        y_lr_reg = reg_net.call_args[0][1]
        assert y_lr_reg.shape[1] == C_LR + date_embedding.shape[1]
        torch.testing.assert_close(
            y_lr_reg[:, C_LR:, 0, 0], date_embedding
        )
        torch.testing.assert_close(
            y_lr_reg, torch.cat([img_lr, date_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)], dim=1)
        )

    def test_diffusion_net_receives_date_embedding(
        self, img_clean, img_lr, date_embedding
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, date_embedding=date_embedding)
        y_lr_diff = diff_net.call_args[0][1]
        assert y_lr_diff.shape[1] == C_LR + date_embedding.shape[1]
        torch.testing.assert_close(
            y_lr_diff[:, C_LR:, 0, 0], date_embedding
        )
        torch.testing.assert_close(
            y_lr_diff, torch.cat([img_lr, date_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)], dim=1)
        )

############################################################################
#                   ResidualLoss — lead time label                         #
############################################################################


class TestResidualLossLeadTime:
    """Lead-time-label behaviour in ResidualLoss."""

    def test_lead_time_label_passed_to_both_nets(
        self, img_clean, img_lr, lead_time_label
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, lead_time_label=lead_time_label)
        # Regression net
        reg_kw = reg_net.call_args[1]
        assert "lead_time_label" in reg_kw
        torch.testing.assert_close(reg_kw["lead_time_label"], lead_time_label)
        # Diffusion net
        diff_kw = diff_net.call_args[1]
        assert "lead_time_label" in diff_kw
        torch.testing.assert_close(diff_kw["lead_time_label"], lead_time_label)

    def test_no_lead_time_label_omitted_from_kwargs(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, lead_time_label=None)
        assert "lead_time_label" not in reg_net.call_args[1]
        assert "lead_time_label" not in diff_net.call_args[1]


############################################################################
#              ResidualLoss — hr_mean_conditioning                         #
############################################################################


class TestResidualLossHrMeanConditioning:
    """High-resolution mean conditioning in ResidualLoss."""

    def test_diffusion_conditioning_includes_mean(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net, hr_mean_conditioning=True)
        loss(diff_net, img_clean, img_lr)
        y_lr_diff = diff_net.call_args[0][1]
        # y_lr should have y_mean (C_HR channels) prepended to img_lr (C_LR channels)
        assert y_lr_diff.shape[1] == C_HR + C_LR
        torch.testing.assert_close(y_lr_diff[:,:C_HR,:,:],torch.zeros_like(y_lr_diff[:,:C_HR,:,:]))
        torch.testing.assert_close(y_lr_diff[:,C_HR:,:,:], img_lr)

    def test_diffusion_conditioning_excludes_mean_when_disabled(
        self, img_clean, img_lr
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net, hr_mean_conditioning=False)
        loss(diff_net, img_clean, img_lr)
        y_lr_diff = diff_net.call_args[0][1]
        assert y_lr_diff.shape[1] == C_LR


############################################################################
#              ResidualLoss — use_patch_grad_acc                           #
############################################################################


class TestResidualLossPatchGradAcc:
    """Test use_patch_grad_acc reuse of cached y_mean."""

    def test_y_mean_cached_after_first_call(self, img_clean, img_lr):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        assert loss.y_mean is None
        loss(diff_net, img_clean, img_lr, use_patch_grad_acc=True)
        assert loss.y_mean is not None

    def test_regression_net_not_called_when_y_mean_cached(
        self, img_clean, img_lr
    ):
        """When use_patch_grad_acc=True and y_mean is already cached,
        the regression net should not be called again."""
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        # First call populates y_mean
        loss(diff_net, img_clean, img_lr, use_patch_grad_acc=True)
        assert reg_net.call_count == 1
        # Second call should reuse y_mean
        loss(diff_net, img_clean, img_lr, use_patch_grad_acc=True)
        assert reg_net.call_count == 1

    def test_regression_net_called_without_patch_grad_acc(
        self, img_clean, img_lr
    ):
        """Without use_patch_grad_acc, regression net is called every time."""
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, use_patch_grad_acc=False)
        loss(diff_net, img_clean, img_lr, use_patch_grad_acc=False)
        assert reg_net.call_count == 2


############################################################################
#                 ResidualLoss — patching integration                      #
############################################################################


class TestResidualLossPatching:
    """Tests for ResidualLoss with RandomPatching2D."""

    @pytest.fixture()
    def patching(self):
        return RandomPatching2D(
            img_shape=(H, W), patch_shape=(32, 32), patch_num=2
        )

    def test_output_shape_with_patching(self, img_clean, img_lr, patching):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(diff_net, img_clean, img_lr, patching=patching)
        # Output should be (B * num_patches, C_HR, patch_H, patch_W)
        assert result.shape[1] == C_HR
        assert result.shape[2] == 32
        assert result.shape[3] == 32
        assert result.shape[0] == B * 2  # More samples due to patching

    def test_diffusion_net_receives_arguments(self, img_clean, img_lr, patching):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        loss(diff_net, img_clean, img_lr, patching=patching)
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        sigma_arg = diff_net.call_args[0][2]
        assert y_arg.shape == (B * 2, C_HR, 32, 32)
        assert y_lr_arg.shape == (B * 2, 2*C_LR, 32, 32)
        assert sigma_arg.shape == (2 * B, 1, 1, 1)

    def test_patching_with_static_channels(
        self, img_clean, img_lr, static_channels, patching
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net)
        result = loss(
            diff_net,
            img_clean,
            img_lr,
            static_channels=static_channels,
            patching=patching,
        )
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        assert y_arg.shape == (B * 2, C_HR, 32, 32)
        assert y_lr_arg.shape == (B * 2, 2*C_LR + 2*static_channels.shape[1], 32, 32)
        assert result.shape == (B * 2, C_HR, 32, 32)

    def test_patching_with_hr_mean_conditioning(
        self, img_clean, img_lr, patching
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(
            regression_net=reg_net, hr_mean_conditioning=True
        )
        result = loss(diff_net, img_clean, img_lr, patching=patching)
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        assert y_arg.shape == (B * 2, C_HR, 32, 32)
        assert y_lr_arg.shape == (B * 2, 2*C_LR+C_HR, 32, 32)
        assert result.shape == (B * 2, C_HR, 32, 32)


############################################################################
#                  ResidualLoss — combined options                         #
############################################################################


class TestResidualLossCombined:
    """Tests with multiple optional arguments combined."""

    def test_all_optional_args_no_patching(
        self,
        img_clean,
        img_lr,
        static_channels,
        date_embedding,
        lead_time_label,
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(
            regression_net=reg_net, hr_mean_conditioning=True
        )
        result = loss(
            diff_net,
            img_clean,
            img_lr,
            static_channels=static_channels,
            date_embedding=date_embedding,
            lead_time_label=lead_time_label,
        )
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        assert y_arg.shape == (B, C_HR, 64, 64)
        assert y_lr_arg.shape == (B, C_LR+C_HR+static_channels.shape[1]+date_embedding.shape[1], 64, 64)
        assert result.shape == img_clean.shape

    def test_all_optional_args_with_patching(
        self,
        img_clean,
        img_lr,
        static_channels,
        date_embedding,
        lead_time_label,
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        patching = RandomPatching2D(img_shape=(H, W), patch_shape=(32, 32), patch_num=2)
        loss = ResidualLoss(
            regression_net=reg_net, hr_mean_conditioning=True
        )
        result = loss(
            diff_net,
            img_clean,
            img_lr,
            static_channels=static_channels,
            date_embedding=date_embedding,
            patching=patching,
            lead_time_label=lead_time_label,
        )
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        assert y_arg.shape == (B * 2, C_HR, 32, 32)
        assert y_lr_arg.shape == (B * 2, 2*C_LR + C_HR + 2*static_channels.shape[1] + date_embedding.shape[1], 32, 32)
        assert result.shape == (B * 2, C_HR, 32, 32)

    def test_augment_with_hr_mean_conditioning_static_and_date(
        self, img_clean, img_lr, static_channels, date_embedding
    ):
        reg_net = _make_dummy_net()
        diff_net = _make_dummy_net()
        loss = ResidualLoss(regression_net=reg_net, hr_mean_conditioning=True)
        result = loss(
            diff_net,
            img_clean,
            img_lr,
            static_channels=static_channels,
            date_embedding=date_embedding,
            augment_pipe=_make_identity_augment(),
        )
        y_arg = diff_net.call_args[0][0]
        y_lr_arg = diff_net.call_args[0][1]
        assert y_arg.shape == (B, C_HR, 64, 64)
        assert y_lr_arg.shape == (B, C_LR+C_HR+static_channels.shape[1]+date_embedding.shape[1], 64, 64)
        assert result.shape == img_clean.shape
        assert (result >= 0).all()
