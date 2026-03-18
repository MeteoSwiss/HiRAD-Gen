# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn

from hirad.models.unet import UNet


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


############################################################################
#                          UNet — __init__                                 #
############################################################################


class TestUNetInitResolution:
    """Test UNet.__init__ resolution handling."""

    @patch("hirad.models.unet.network_module")
    def test_int_resolution_sets_square(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=128, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.img_shape_x == 128
        assert unet.img_shape_y == 128

    @patch("hirad.models.unet.network_module")
    def test_tuple_resolution_sets_height_width(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=(96, 128), img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.img_shape_y == 96
        assert unet.img_shape_x == 128

    @patch("hirad.models.unet.network_module")
    def test_stores_channel_counts(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.img_in_channels == C_IN
        assert unet.img_out_channels == C_OUT


class TestUNetInitModelType:
    """Test that UNet creates the correct underlying model type."""

    @patch("hirad.models.unet.network_module")
    def test_default_model_type_is_song_unet_pos_embd(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        mock_cls.assert_called_once()

    @patch("hirad.models.unet.network_module")
    def test_custom_model_type_song_unet(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNet = mock_cls
        UNet(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            model_type="SongUNet",
        )
        mock_cls.assert_called_once()

    @patch("hirad.models.unet.network_module")
    def test_model_receives_combined_in_channels(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["in_channels"] == C_IN + C_OUT

    @patch("hirad.models.unet.network_module")
    def test_model_receives_out_channels(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["out_channels"] == C_OUT

    @patch("hirad.models.unet.network_module")
    def test_extra_kwargs_forwarded_to_model(self, mock_module):
        mock_cls = MagicMock()
        mock_module.SongUNetPosEmbd = mock_cls
        UNet(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            model_channels=256,
            num_blocks=8,
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model_channels"] == 256
        assert call_kwargs["num_blocks"] == 8


############################################################################
#                      UNet — use_fp16 property                            #
############################################################################


class TestUNetUseFp16:
    """Test use_fp16 property getter and setter."""

    @patch("hirad.models.unet.network_module")
    def test_default_fp16_is_false(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.use_fp16 is False

    @patch("hirad.models.unet.network_module")
    def test_set_fp16_true(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(
            img_resolution=64,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            use_fp16=True,
        )
        assert unet.use_fp16 is True

    @patch("hirad.models.unet.network_module")
    def test_set_fp16_via_setter(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        unet.use_fp16 = True
        assert unet.use_fp16 is True
        unet.use_fp16 = False
        assert unet.use_fp16 is False

    @patch("hirad.models.unet.network_module")
    def test_set_fp16_accepts_int_0_and_1(self, mock_module):
        """Older checkpoints may store 0/1 instead of bool."""
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        unet.use_fp16 = 1
        assert unet.use_fp16 == 1
        unet.use_fp16 = 0
        assert unet.use_fp16 == 0

    @patch("hirad.models.unet.network_module")
    def test_set_fp16_invalid_type_raises(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        with pytest.raises(ValueError, match="must be a boolean"):
            unet.use_fp16 = "yes"

    @patch("hirad.models.unet.network_module")
    def test_set_fp16_none_raises(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        with pytest.raises(ValueError, match="must be a boolean"):
            unet.use_fp16 = None


############################################################################
#                       UNet — forward                                     #
############################################################################


class TestUNetForwardBasic:
    """Basic forward pass tests for UNet."""

    @patch("hirad.models.unet.network_module")
    def test_output_shape(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        out = unet(x, lr)
        torch.testing.assert_close(out, torch.zeros((B, C_OUT, H, W)))

    @patch("hirad.models.unet.network_module")
    def test_output_dtype_is_float32(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        out = unet(x, lr)
        assert out.dtype == torch.float32

    @patch("hirad.models.unet.network_module")
    def test_concatenates_x_and_img_lr(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.ones(B, C_OUT, H, W)
        lr = torch.ones(B, C_IN, H, W) * 2
        unet(x, lr)
        model_input = mock_model.call_args[0][0]
        assert model_input.shape[1] == C_OUT + C_IN
        # First channels should be x, remaining should be img_lr
        torch.testing.assert_close(model_input[:, :C_OUT], x)
        torch.testing.assert_close(model_input[:, C_OUT:], lr)

    @patch("hirad.models.unet.network_module")
    def test_model_receives_zero_sigma(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        unet(x, lr)
        sigma_arg = mock_model.call_args[0][1]
        torch.testing.assert_close(
            sigma_arg, torch.zeros(B, dtype=torch.float32)
        )

    @patch("hirad.models.unet.network_module")
    def test_model_receives_none_class_labels(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        unet(x, lr)
        call_kwargs = mock_model.call_args[1]
        assert call_kwargs["class_labels"] is None

    @patch("hirad.models.unet.network_module")
    def test_kwargs_forwarded_to_model(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        lead_time = torch.randint(49, size=(B,))
        unet(x, lr, lead_time_label=lead_time)
        call_kwargs = mock_model.call_args[1]
        assert "lead_time_label" in call_kwargs
        torch.testing.assert_close(call_kwargs["lead_time_label"], lead_time)


class TestUNetForwardImgLrNone:
    """Test forward pass when img_lr is None."""

    @patch("hirad.models.unet.network_module")
    def test_no_concatenation_when_img_lr_none(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        unet(x, img_lr=None)
        model_input = mock_model.call_args[0][0]
        # Without img_lr, input should only be x
        assert model_input.shape[1] == C_OUT


class TestUNetForwardDtypeValidation:
    """Test dtype enforcement in forward pass."""

    @patch("hirad.models.unet.network_module")
    def test_raises_on_dtype_mismatch(self, mock_module):
        """Model should raise if the underlying model returns wrong dtype."""
        mock_model = MagicMock(spec=nn.Module)
        # Return fp16 when fp32 is expected
        mock_model.side_effect = lambda x, sigma, class_labels=None, **kw: torch.zeros(
            x.shape[0], C_OUT, x.shape[2], x.shape[3], dtype=torch.float16
        )
        mock_model.modules.return_value = iter([])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=H, img_in_channels=C_IN, img_out_channels=C_OUT)
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        with pytest.raises(ValueError, match="Expected the dtype"):
            unet(x, lr)


class TestUNetForwardForceFp32:
    """Test the force_fp32 flag."""

    @patch("hirad.models.unet.network_module")
    def test_force_fp32_uses_float32(self, mock_module):
        mock_model = _make_mock_model()
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(
            img_resolution=H,
            img_in_channels=C_IN,
            img_out_channels=C_OUT,
            use_fp16=True,
        )
        x = torch.zeros(B, C_OUT, H, W)
        lr = torch.randn(B, C_IN, H, W)
        unet(x, lr, force_fp32=True)
        model_input = mock_model.call_args[0][0]
        assert model_input.dtype == torch.float32


############################################################################
#                       UNet — round_sigma                                 #
############################################################################


class TestUNetRoundSigma:
    """Test round_sigma method."""

    @patch("hirad.models.unet.network_module")
    def test_float_input(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        result = unet.round_sigma(0.5)
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(0.5)

    @patch("hirad.models.unet.network_module")
    def test_list_input(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        result = unet.round_sigma([0.1, 0.5, 1.0])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        torch.testing.assert_close(result, torch.tensor([0.1, 0.5, 1.0]))

    @patch("hirad.models.unet.network_module")
    def test_tensor_input(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        sigma = torch.tensor([0.2, 0.8])
        result = unet.round_sigma(sigma)
        torch.testing.assert_close(result, sigma)

    @patch("hirad.models.unet.network_module")
    def test_zero_input(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        result = unet.round_sigma(0.0)
        assert result.item() == pytest.approx(0.0)


############################################################################
#                       UNet — amp_mode property                           #
############################################################################


class TestUNetAmpMode:
    """Test amp_mode property getter and setter."""

    @patch("hirad.models.unet.network_module")
    def test_amp_mode_returns_none_when_model_lacks_attr(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        # Remove amp_mode from mock so hasattr returns False
        del mock_model.amp_mode
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.amp_mode is None

    @patch("hirad.models.unet.network_module")
    def test_amp_mode_returns_model_value(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = True
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert unet.amp_mode is True

    @patch("hirad.models.unet.network_module")
    def test_amp_mode_setter_updates_model(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = False
        sub_module = MagicMock()
        sub_module.amp_mode = False
        mock_model.modules.return_value = iter([mock_model, sub_module])
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        unet.amp_mode = True
        assert mock_model.amp_mode is True
        assert sub_module.amp_mode is True

    @patch("hirad.models.unet.network_module")
    def test_amp_mode_setter_rejects_non_bool(self, mock_module):
        mock_model = MagicMock(spec=nn.Module)
        mock_model.amp_mode = False
        mock_module.SongUNetPosEmbd = MagicMock(return_value=mock_model)
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        with pytest.raises(TypeError, match="amp_mode must be a boolean"):
            unet.amp_mode = "yes"


############################################################################
#            UNet — _backward_compat_arg_mapper                            #
############################################################################


# class TestUNetBackwardCompat:
#     """Test _backward_compat_arg_mapper for version-based argument migration."""

#     def test_v010_removes_img_channels(self):
#         args = {
#             "img_resolution": 64,
#             "img_in_channels": C_IN,
#             "img_out_channels": C_OUT,
#             "img_channels": 10,
#         }
#         result = UNet._backward_compat_arg_mapper("0.1.0", args)
#         assert "img_channels" not in result

#     def test_v010_removes_sigma_params(self):
#         args = {
#             "img_resolution": 64,
#             "img_in_channels": C_IN,
#             "img_out_channels": C_OUT,
#             "sigma_min": 0.002,
#             "sigma_max": 80.0,
#             "sigma_data": 0.5,
#         }
#         result = UNet._backward_compat_arg_mapper("0.1.0", args)
#         assert "sigma_min" not in result
#         assert "sigma_max" not in result
#         assert "sigma_data" not in result

#     def test_v010_keeps_valid_args(self):
#         args = {
#             "img_resolution": 64,
#             "img_in_channels": C_IN,
#             "img_out_channels": C_OUT,
#         }
#         result = UNet._backward_compat_arg_mapper("0.1.0", args)
#         assert result["img_resolution"] == 64
#         assert result["img_in_channels"] == C_IN
#         assert result["img_out_channels"] == C_OUT

#     def test_non_v010_preserves_all_args(self):
#         args = {
#             "img_resolution": 64,
#             "img_in_channels": C_IN,
#             "img_out_channels": C_OUT,
#             "img_channels": 10,
#             "sigma_min": 0.002,
#         }
#         result = UNet._backward_compat_arg_mapper("0.2.0", args)
#         assert "img_channels" in result
#         assert "sigma_min" in result


############################################################################
#                    UNet — nn.Module integration                          #
############################################################################


class TestUNetModuleIntegration:
    """Test that UNet behaves as a proper nn.Module."""

    @patch("hirad.models.unet.network_module")
    def test_is_nn_module(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert isinstance(unet, nn.Module)

    @patch("hirad.models.unet.network_module")
    def test_model_attribute_exists(self, mock_module):
        mock_module.SongUNetPosEmbd = MagicMock()
        unet = UNet(img_resolution=64, img_in_channels=C_IN, img_out_channels=C_OUT)
        assert hasattr(unet, "model")
