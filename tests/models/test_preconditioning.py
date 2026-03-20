# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from hirad.models.preconditioning import EDMPrecondSuperResolution


# ---------------------------------------------------------------------------
#  Helpers / fixtures
# ---------------------------------------------------------------------------

B, C_IN, C_OUT, H, W = 2, 4, 3, 64, 64


def _make_mock_model(out_channels=C_OUT):
    """Return a MagicMock that behaves like a SongUNet-style model."""
    model = MagicMock(spec=nn.Module)
    model.side_effect = lambda x, sigma, class_labels=None, **kw: torch.zeros(
        x.shape[0], out_channels, x.shape[2], x.shape[3],
        dtype=x.dtype, device=x.device,
    )
    model.modules.return_value = iter([])
    return model


@pytest.fixture()
def img_x():
    return torch.randn(B, C_OUT, H, W)


@pytest.fixture()
def img_lr():
    return torch.randn(B, C_IN, H, W)


@pytest.fixture()
def sigma():
    return torch.ones(B) * 0.5


############################################################################
#                   EDMPrecondSuperResolution — __init__                   #
############################################################################


class TestEDMInitResolution:
    """Test EDMPrecondSuperResolution.__init__ resolution handling."""

    @patch("hirad.models.preconditioning.network_module")
    def test_int_resolution_stored(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=128, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.img_resolution == 128

    @patch("hirad.models.preconditioning.network_module")
    def test_tuple_resolution_stored(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=(96, 128), img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.img_resolution == (96, 128)

    @patch("hirad.models.preconditioning.network_module")
    def test_stores_channel_counts(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.img_in_channels == C_IN
        assert edm.img_out_channels == C_OUT


class TestEDMInitModelType:
    """Test that EDMPrecondSuperResolution creates the correct underlying model type."""

    @patch("hirad.models.preconditioning.network_module")
    def test_default_model_type_is_song_unet_pos_embd(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        mock_cls.assert_called_once()

    @patch("hirad.models.preconditioning.network_module")
    def test_custom_model_type_song_unet(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNet = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            model_type="SongUNet",
        )
        mock_cls.assert_called_once()

    @patch("hirad.models.preconditioning.network_module")
    def test_model_receives_img_resolution(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=(80, 120), img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["img_resolution"] == (80, 120)

    @patch("hirad.models.preconditioning.network_module")
    def test_model_receives_combined_in_channels(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["in_channels"] == C_IN + C_OUT

    @patch("hirad.models.preconditioning.network_module")
    def test_model_receives_out_channels(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["out_channels"] == C_OUT

    @patch("hirad.models.preconditioning.network_module")
    def test_extra_kwargs_forwarded_to_model(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        EDMPrecondSuperResolution(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            model_channels=256,
            num_blocks=8,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_channels"] == 256
        assert call_kwargs["num_blocks"] == 8

    @patch("hirad.models.preconditioning.network_module")
    def test_model_attribute_exists(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert hasattr(edm, "model")


class TestEDMInitSigmaDefaults:
    """Test sigma-related default values."""

    @patch("hirad.models.preconditioning.network_module")
    def test_default_sigma_data(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.sigma_data == 0.5

    @patch("hirad.models.preconditioning.network_module")
    def test_default_sigma_min(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.sigma_min == 0.0

    @patch("hirad.models.preconditioning.network_module")
    def test_default_sigma_max(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.sigma_max == float("inf")

    @patch("hirad.models.preconditioning.network_module")
    def test_custom_sigma_info(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            sigma_data=1.0,
            sigma_min=0.002,
            sigma_max=80.0,
        )
        assert edm.sigma_data == 1.0
        assert edm.sigma_min == 0.002
        assert edm.sigma_max == 80.0


class TestEDMInitFp16:
    """Test use_fp16 stored correctly at init."""

    @patch("hirad.models.preconditioning.network_module")
    def test_default_fp16_is_false(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.use_fp16 is False

    @patch("hirad.models.preconditioning.network_module")
    def test_set_fp16_true(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            use_fp16=True,
        )
        assert edm.use_fp16 is True


############################################################################
#                EDMPrecondSuperResolution — _scaling_fn                   #
############################################################################


class TestEDMScalingFn:
    """Test the static _scaling_fn method."""

    def test_output_shape(self):
        x = torch.randn(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        c_in = torch.ones(B, 1, 1, 1) * 0.5
        result = EDMPrecondSuperResolution._scaling_fn(x, lr, c_in)
        assert result.shape == (B, C_OUT + C_IN, H, W)

    def test_first_channels_are_scaled_x(self):
        x = torch.ones(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        c_in = torch.ones(B, 1, 1, 1) * 2.0
        result = EDMPrecondSuperResolution._scaling_fn(x, lr, c_in)
        torch.testing.assert_close(result[:, :C_OUT], x * 2.0)

    def test_last_channels_are_unscaled_lr(self):
        x = torch.randn(B, C_OUT, H, W)
        lr = torch.ones(B, C_IN, H, W) * 3.0
        c_in = torch.ones(B, 1, 1, 1) * 0.5
        result = EDMPrecondSuperResolution._scaling_fn(x, lr, c_in)
        torch.testing.assert_close(result[:, C_OUT:], lr)

    def test_lr_cast_to_x_dtype(self):
        x = torch.randn(B, C_OUT, H, W, dtype=torch.float32)
        lr = torch.randn(B, C_IN, H, W, dtype=torch.float64)
        c_in = torch.ones(B, 1, 1, 1)
        result = EDMPrecondSuperResolution._scaling_fn(x, lr, c_in)
        assert result.dtype == torch.float32


############################################################################
#                EDMPrecondSuperResolution — forward                       #
############################################################################


class TestEDMForwardBasic:
    """Basic forward pass tests for EDMPrecondSuperResolution."""

    @patch("hirad.models.preconditioning.network_module")
    def test_output_shape(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        out = edm(img_x, img_lr, sigma)
        assert out.shape == (B, C_OUT, H, W)

    @patch("hirad.models.preconditioning.network_module")
    def test_output_dtype_is_float32(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        out = edm(img_x, img_lr, sigma)
        assert out.dtype == torch.float32

    @patch("hirad.models.preconditioning.network_module")
    def test_model_input_has_combined_channels(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        edm(img_x, img_lr, sigma)
        model_input = mock_model.call_args[0][0]
        assert model_input.shape[1] == C_OUT + C_IN

    @patch("hirad.models.preconditioning.network_module")
    def test_model_receives_flattened_c_noise(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        edm(img_x, img_lr, sigma)
        c_noise_arg = mock_model.call_args[0][1]
        assert c_noise_arg.ndim == 1
        assert c_noise_arg.shape[0] == B

    @patch("hirad.models.preconditioning.network_module")
    def test_model_receives_none_class_labels(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        edm(img_x, img_lr, sigma)
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["class_labels"] is None

    @patch("hirad.models.preconditioning.network_module")
    def test_kwargs_forwarded_to_model(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        lead_time = torch.randint(49, size=(B,))
        edm(img_x, img_lr, sigma, lead_time_label=lead_time)
        call_kwargs = mock_model.call_args[1]
        assert "lead_time_label" in call_kwargs
        torch.testing.assert_close(call_kwargs["lead_time_label"], lead_time)


class TestEDMForwardPreconditioning:
    """Test that the EDM preconditioning coefficients are applied correctly."""

    @patch("hirad.models.preconditioning.network_module")
    def test_c_noise_is_log_sigma_over_4(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        sigma_val = 2.0
        sigma = torch.full((B,), sigma_val)
        x = torch.randn(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        edm(x, lr, sigma)
        c_noise_arg = mock_model.call_args[0][1]
        expected = torch.full((B,), torch.tensor(sigma_val).log().item() / 4)
        torch.testing.assert_close(c_noise_arg, expected)

    @patch("hirad.models.preconditioning.network_module")
    def test_output_is_c_skip_x_plus_c_out_F_x(self, mock_module):
        """D(x) = c_skip * x + c_out * F(x); with F(x)=0, D(x) = c_skip * x."""
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        sigma_data = 0.5
        edm = EDMPrecondSuperResolution(
            img_resolution=H,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            sigma_data=sigma_data,
        )
        sigma_val = 1.0
        sigma = torch.full((B,), sigma_val)
        x = torch.ones(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        out = edm(x, lr, sigma)
        # Since mock model returns zeros, D(x) = c_skip * x
        c_skip = sigma_data**2 / (sigma_val**2 + sigma_data**2)
        expected = torch.full((B, C_OUT, H, W), c_skip)
        torch.testing.assert_close(out, expected)

    @patch("hirad.models.preconditioning.network_module")
    def test_sigma_reshaped_to_4d(self, mock_module, img_x, img_lr):
        """Sigma with shape (B,) should be reshaped to (B, 1, 1, 1)."""
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        sigma_1d = torch.ones(B)
        # Should not raise
        edm(img_x, img_lr, sigma_1d)

    @patch("hirad.models.preconditioning.network_module")
    def test_sigma_2d_reshaped_to_4d(self, mock_module, img_x, img_lr):
        """Sigma with shape (B, 1) should be reshaped to (B, 1, 1, 1)."""
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        sigma_2d = torch.ones(B, 1)
        # Should not raise
        edm(img_x, img_lr, sigma_2d)

    @patch("hirad.models.preconditioning.network_module")
    def test_sigma_4d_accepted(self, mock_module, img_x, img_lr):
        """Sigma with shape (B, 1, 1, 1) should be accepted."""
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        sigma_4d = torch.ones(B, 1, 1, 1)
        # Should not raise
        edm(img_x, img_lr, sigma_4d)


class TestEDMForwardImgLrNone:
    """Test forward pass when img_lr is None."""

    @patch("hirad.models.preconditioning.network_module")
    def test_no_concatenation_when_img_lr_none(self, mock_module, img_x, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        edm(img_x, img_lr=None, sigma=sigma)
        model_input = mock_model.call_args[0][0]
        # Without img_lr, input is c_in * x only
        assert model_input.shape[1] == C_OUT


class TestEDMForwardDtypeValidation:
    """Test dtype enforcement in forward pass."""

    @patch("hirad.models.preconditioning.network_module")
    def test_raises_on_dtype_mismatch(self, mock_module, img_x, img_lr, sigma):
        """Model should raise if the underlying model returns wrong dtype."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.side_effect = lambda x, sigma, class_labels=None, **kw: torch.zeros(
            x.shape[0], C_OUT, x.shape[2], x.shape[3], dtype=torch.float16,
        )
        mock_model.modules.return_value = iter([])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        with pytest.raises(ValueError, match="Expected the dtype"):
            edm(img_x, img_lr, sigma)


class TestEDMForwardForceFp32:
    """Test the force_fp32 flag."""

    #TODO: Test doesn't make sence when device is cpu.
    @patch("hirad.models.preconditioning.network_module")
    def test_force_fp32_uses_float32(self, mock_module, img_x, img_lr, sigma):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            use_fp16=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        edm(img_x.to(device), img_lr.to(device), sigma.to(device), force_fp32=True)
        model_input = mock_model.call_args[0][0]
        assert model_input.dtype == torch.float32


class TestEDMForwardAutocastEnabled:
    """Test that dtype validation is skipped when autocast is enabled."""

    @patch("hirad.models.preconditioning.network_module")
    def test_no_dtype_check_when_autocast_enabled(self, mock_module, img_x, img_lr, sigma):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.side_effect = lambda x, sigma, class_labels=None, **kw: torch.zeros(
            x.shape[0], C_OUT, x.shape[2], x.shape[3], dtype=torch.float16,
        )
        mock_model.modules.return_value = iter([])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        with torch.autocast("cuda"):
            out = edm(img_x, img_lr, sigma)
            assert out.dtype == torch.float32


############################################################################
#             EDMPrecondSuperResolution — round_sigma                      #
############################################################################


class TestEDMRoundSigma:
    """Test round_sigma static method."""

    def test_float_input(self):
        result = EDMPrecondSuperResolution.round_sigma(0.5)
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(0.5)

    def test_list_input(self):
        result = EDMPrecondSuperResolution.round_sigma([0.1, 0.5, 1.0])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        torch.testing.assert_close(result, torch.tensor([0.1, 0.5, 1.0]))

    def test_tensor_input(self):
        sigma = torch.tensor([0.2, 0.8])
        result = EDMPrecondSuperResolution.round_sigma(sigma)
        torch.testing.assert_close(result, sigma)


############################################################################
#             EDMPrecondSuperResolution — amp_mode property                #
############################################################################


class TestEDMAmpMode:
    """Test amp_mode property getter and setter."""

    @patch("hirad.models.preconditioning.network_module")
    def test_amp_mode_returns_none_when_model_lacks_attr(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        del mock_model.amp_mode
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.amp_mode is None

    @patch("hirad.models.preconditioning.network_module")
    def test_amp_mode_returns_model_value(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = True
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.amp_mode is True

    @patch("hirad.models.preconditioning.network_module")
    def test_amp_mode_setter_updates_model_and_submodules(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = False
        sub_module = MagicMock()
        sub_module.amp_mode = False
        mock_model.modules.return_value = iter([sub_module])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        edm.amp_mode = True
        assert mock_model.amp_mode is True
        assert sub_module.amp_mode is True

    @patch("hirad.models.preconditioning.network_module")
    def test_amp_mode_setter_rejects_non_bool(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = False
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        with pytest.raises(TypeError, match="amp_mode must be a boolean"):
            edm.amp_mode = "yes"

    @patch("hirad.models.preconditioning.network_module")
    def test_amp_mode_setter_skips_model_without_attr(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        del mock_model.amp_mode
        mock_model.modules.return_value = iter([])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        # Should not raise even when model lacks amp_mode
        edm.amp_mode = True


############################################################################
#             EDMPrecondSuperResolution — nn.Module integration            #
############################################################################


class TestEDMModuleIntegration:
    """Test that EDMPrecondSuperResolution behaves as a proper nn.Module."""

    @patch("hirad.models.preconditioning.network_module")
    def test_is_nn_module(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert isinstance(edm, nn.Module)

    @patch("hirad.models.preconditioning.network_module")
    def test_scaling_fn_attribute_set(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        edm = EDMPrecondSuperResolution(
            img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT,
        )
        assert edm.scaling_fn is EDMPrecondSuperResolution._scaling_fn
