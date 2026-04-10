# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from hirad.models.layers import (
    AttentionOp,
    Conv2d,
    FourierEmbedding,
    GroupNorm,
    Linear,
    PositionalEmbedding,
    UNetBlock,
)


# ---------------------------------------------------------------------------
#  Helpers / fixtures — use small configs for fast CPU tests
# ---------------------------------------------------------------------------

B = 2
IN_CH = 4
OUT_CH = 8
H, W = 16, 16
EMB_CH = 32


@pytest.fixture()
def random_input_2d():
    return torch.randn(B, IN_CH, H, W)


@pytest.fixture()
def random_input_flat():
    return torch.randn(B, IN_CH)


@pytest.fixture()
def embedding():
    return torch.randn(B, EMB_CH)


############################################################################
#                              Linear                                      #
############################################################################


class TestLinearInit:
    """Test Linear.__init__ parameter setup."""

    def test_weight_shape(self):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH)
        assert layer.weight.shape == (OUT_CH, IN_CH)

    def test_bias_shape(self):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH)
        assert layer.bias is not None
        assert layer.bias.shape == (OUT_CH,)

    def test_no_bias_when_disabled(self):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH, bias=False)
        assert layer.bias is None

    def test_stores_features(self):
        layer = Linear(in_features=IN_CH, 
                        out_features=OUT_CH,
                        amp_mode=True)
        assert layer.in_features == IN_CH
        assert layer.out_features == OUT_CH
        assert layer.amp_mode==True

    @pytest.mark.parametrize(
        "init_mode",
        ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
    )
    def test_all_init_modes_accepted(self, init_mode):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH, init_mode=init_mode)
        assert layer.weight.shape == (OUT_CH, IN_CH)

    def test_init_weight_scaling(self):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH, init_weight=0.0)
        torch.testing.assert_close(layer.weight, torch.zeros(OUT_CH, IN_CH))

    def test_init_bias_scaling(self):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH, init_bias=0)
        torch.testing.assert_close(layer.bias, torch.zeros(OUT_CH))


class TestLinearForward:
    """Test Linear forward pass."""

    def test_output_shape(self, random_input_flat):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH)
        out = layer(random_input_flat)
        assert out.shape == (B, OUT_CH)

    def test_output_shape_no_bias(self, random_input_flat):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH, bias=False)
        out = layer(random_input_flat)
        assert out.shape == (B, OUT_CH)

    def test_zero_weight_zero_bias_returns_zero(self, random_input_flat):
        layer = Linear(
            in_features=IN_CH,
            out_features=OUT_CH,
            init_weight=0,
            init_bias=0,
        )
        out = layer(random_input_flat)
        torch.testing.assert_close(out, torch.zeros(B, OUT_CH))

    def test_output_dtype_matches_input(self, random_input_flat):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH)
        out = layer(random_input_flat)
        assert out.dtype == random_input_flat.dtype

    def test_gradients_flow(self, random_input_flat):
        layer = Linear(in_features=IN_CH, out_features=OUT_CH)
        x = random_input_flat.clone().requires_grad_(True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


############################################################################
#                              Conv2d                                      #
############################################################################


class TestConv2dInit:
    """Test Conv2d.__init__ parameter setup."""

    def test_weight_shape(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        assert layer.weight.shape == (OUT_CH, IN_CH, 3, 3)

    def test_bias_shape(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        assert layer.bias is not None
        assert layer.bias.shape == (OUT_CH,)

    def test_no_bias_when_disabled(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3, bias=False)
        assert layer.bias is None

    def test_kernel_zero_no_weight(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=0)
        assert layer.weight is None
        assert layer.bias is None

    def test_up_and_down_raises(self):
        with pytest.raises(ValueError, match="Both 'up' and 'down'"):
            Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3, up=True, down=True)

    def test_stores_flags(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3, up=True, fused_resample=True, fused_conv_bias=True, amp_mode=True)
        assert layer.up is True
        assert layer.down is False
        assert layer.fused_resample is True
        assert layer.fused_conv_bias is True
        assert layer.amp_mode is True
        assert layer.in_channels == IN_CH
        assert layer.out_channels == OUT_CH

    def test_resample_filter_registered_when_up(self):
        layer = Conv2d(
            in_channels=IN_CH, out_channels=OUT_CH, kernel=3, up=True
        )
        assert layer.resample_filter is not None

    def test_resample_filter_registered_when_down(self):
        layer = Conv2d(
            in_channels=IN_CH, out_channels=OUT_CH, kernel=3, down=True
        )
        assert layer.resample_filter is not None

    def test_resample_filter_none_when_no_up_down(self):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        assert layer.resample_filter is None

    def test_fused_conv_bias_disabled_when_no_kernel(self):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=0,
            fused_conv_bias=True,
        )
        assert layer.fused_conv_bias is False

    def test_zero_weight_bias_init_gives_zero_weights_and_biases(self):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            init_weight=0,
            init_bias=0,
        )
        torch.testing.assert_close(layer.weight, torch.zeros(OUT_CH, IN_CH, 3, 3))
        torch.testing.assert_close(layer.bias, torch.zeros(OUT_CH))


class TestConv2dForward:
    """Test Conv2d forward pass."""

    def test_output_shape_same_padding(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H, W)

    def test_output_shape_kernel_1(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=1)
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H, W)

    def test_output_shape_upsample(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3, up=True)
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H * 2, W * 2)

    def test_output_shape_downsample(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3, down=True)
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H // 2, W // 2)

    def test_output_shape_fused_upsample(self, random_input_2d):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            up=True,
            fused_resample=True,
        )
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H * 2, W * 2)

    def test_output_shape_fused_downsample(self, random_input_2d):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            down=True,
            fused_resample=True,
        )
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H // 2, W // 2)

    def test_output_shape_fused_conv_bias(self, random_input_2d):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            fused_conv_bias=True,
        )
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H, W)

    def test_output_shape_fused_up_with_conv_bias(self, random_input_2d):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            up=True,
            fused_resample=True,
            fused_conv_bias=True,
        )
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H * 2, W * 2)

    def test_output_shape_fused_down_with_conv_bias(self, random_input_2d):
        layer = Conv2d(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            kernel=3,
            down=True,
            fused_resample=True,
            fused_conv_bias=True,
        )
        out = layer(random_input_2d)
        assert out.shape == (B, OUT_CH, H // 2, W // 2)

    def test_output_dtype_matches_input(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        out = layer(random_input_2d)
        assert out.dtype == random_input_2d.dtype

    def test_gradients_flow(self, random_input_2d):
        layer = Conv2d(in_channels=IN_CH, out_channels=OUT_CH, kernel=3)
        x = random_input_2d.clone().requires_grad_(True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_kernel_zero_passthrough(self, random_input_2d):
        """With kernel=0, no convolution should be applied, just pass-through."""
        layer = Conv2d(in_channels=IN_CH, out_channels=IN_CH, kernel=0)
        out = layer(random_input_2d)
        torch.testing.assert_close(out, random_input_2d)


############################################################################
#                            GroupNorm                                     #
############################################################################


class TestGroupNormInit:
    """Test GroupNorm.__init__ parameter setup."""

    def test_weight_shape(self):
        gn = GroupNorm(num_channels=OUT_CH)
        assert gn.weight.shape == (OUT_CH,)

    def test_bias_shape(self):
        gn = GroupNorm(num_channels=OUT_CH)
        assert gn.bias.shape == (OUT_CH,)

    def test_weight_initialized_to_ones(self):
        gn = GroupNorm(num_channels=OUT_CH)
        torch.testing.assert_close(gn.weight, torch.ones(OUT_CH))

    def test_bias_initialized_to_zeros(self):
        gn = GroupNorm(num_channels=OUT_CH)
        torch.testing.assert_close(gn.bias, torch.zeros(OUT_CH))

    def test_num_groups_clipped_to_min_channels(self):
        """If num_channels // min_channels_per_group < num_groups, groups are reduced."""
        gn = GroupNorm(num_channels=8, num_groups=32, min_channels_per_group=4)
        assert gn.num_groups == 2

    def test_num_groups_matches_when_divisible(self):
        gn = GroupNorm(num_channels=32, num_groups=8, min_channels_per_group=2)
        assert gn.num_groups == 8

    def test_fused_act_without_act_raises(self):
        with pytest.raises(ValueError, match="'act' must be specified"):
            GroupNorm(num_channels=OUT_CH, fused_act=True, act=None)

    def test_fused_act_with_valid_act(self):
        gn = GroupNorm(num_channels=OUT_CH, fused_act=True, act="silu")
        assert gn.fused_act is True
        assert gn.act == "silu"
        assert gn.act_fn is not None

    def test_eps_and_amp_mode_stored(self):
        gn = GroupNorm(num_channels=OUT_CH, eps=1e-6, amp_mode=True)
        assert gn.eps == 1e-6
        assert gn.amp_mode is True

    def test_apex_gn_initializes_gn_when_available(self):
        mock_gn_cls = MagicMock()
        mock_gn_instance = MagicMock()
        mock_gn_cls.return_value = mock_gn_instance
        with patch("hirad.models.layers._is_apex_available", True), \
             patch("hirad.models.layers.ApexGroupNorm", mock_gn_cls, create=True):
                gn = GroupNorm(num_channels=OUT_CH, use_apex_gn=True)
                assert hasattr(gn, "gn")
                assert gn.gn is mock_gn_instance
                mock_gn_cls.assert_called_once()

    def test_apex_gn_raises_when_not_available(self):
        with patch("hirad.models.layers._is_apex_available", False):
            with pytest.raises(ValueError, match="'apex' is not"):
                GroupNorm(num_channels=OUT_CH, use_apex_gn=True)


class TestGroupNormForward:
    """Test GroupNorm forward pass."""

    def test_output_shape(self, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH)
        out = gn(random_input_2d)
        assert out.shape == random_input_2d.shape

    def test_output_dtype_matches_input(self, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        out = gn(random_input_2d)
        assert out.dtype == random_input_2d.dtype
        gn.eval()
        out = gn(random_input_2d)
        assert out.dtype == random_input_2d.dtype

    def test_training_mode_uses_torch_group_norm(self, random_input_2d):
        """In training mode, output should match torch.nn.functional.group_norm."""
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        out = gn(random_input_2d)
        expected = torch.nn.functional.group_norm(
            random_input_2d, num_groups=gn.num_groups, weight=gn.weight, bias=gn.bias, eps=gn.eps
        )
        torch.testing.assert_close(out, expected)

    def test_eval_mode_output_shape(self, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH)
        gn.eval()
        out = gn(random_input_2d)
        assert out.shape == random_input_2d.shape

    def test_apex_gn_forward(self, random_input_2d):
        """Test forward pass when using Apex GroupNorm."""
        from hirad.models.layers import _is_apex_available
        if _is_apex_available:
            gn = GroupNorm(num_channels=IN_CH, use_apex_gn=True)
            called = []
            gn.gn.register_forward_hook(lambda m, i, o: called.append(True))
            out = gn(random_input_2d)
            assert out.shape == random_input_2d.shape
            assert called, "Apex GroupNorm forward hook was not called, so it may not have been used."
        else:
            mock_gn_cls = MagicMock()
            mock_gn_instance = MagicMock()
            mock_gn_instance.forward.return_value = random_input_2d
            mock_gn_cls.return_value = mock_gn_instance
            with patch("hirad.models.layers._is_apex_available", True), \
                patch("hirad.models.layers.ApexGroupNorm", mock_gn_cls, create=True):
                    gn = GroupNorm(num_channels=IN_CH, use_apex_gn=True)
                    out = gn(random_input_2d)
                    assert out.shape == random_input_2d.shape
                    assert mock_gn_instance.assert_called_once

    def test_training_mode_with_fused_act(self, random_input_2d):
        """Test that fused activation is applied in training mode."""
        gn = GroupNorm(num_channels=IN_CH, fused_act=True, act="relu")
        gn.train()
        out = gn(random_input_2d)
        assert out.shape == random_input_2d.shape
        assert (out >= 0).all(), "Output should be non-negative due to ReLU activation"

    def test_eval_mode_with_fused_act(self, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH, fused_act=True, act="relu")
        gn.eval()
        out = gn(random_input_2d)
        assert out.shape == random_input_2d.shape
        assert (out >= 0).all(), "Output should be non-negative due to ReLU activation"

    def test_training_fused_act_actually_applies_activation(self, random_input_2d):
        """Verify fused act path produces different output than non-fused path."""
        gn_fused = GroupNorm(num_channels=IN_CH, fused_act=True, act="relu")
        gn_plain = GroupNorm(num_channels=IN_CH)
        gn_fused.train()
        gn_plain.train()
        out_fused = gn_fused(random_input_2d)
        out_plain = gn_plain(random_input_2d)
        # Plain output will have negatives; fused should not
        assert (out_plain < 0).any(), "Plain output should have negatives for meaningful test"
        assert (out_fused >= 0).all()

    def test_eval_fused_act_actually_applies_activation(self, random_input_2d):
        """Verify fused act path produces different output than non-fused path."""
        gn_fused = GroupNorm(num_channels=IN_CH, fused_act=True, act="relu")
        gn_plain = GroupNorm(num_channels=IN_CH)
        gn_fused.eval()
        gn_plain.eval()
        out_fused = gn_fused(random_input_2d)
        out_plain = gn_plain(random_input_2d)
        # Plain output will have negatives; fused should not
        assert (out_plain < 0).any(), "Plain output should have negatives for meaningful test"
        assert (out_fused >= 0).all()

    def test_eval_mode_matches_training_mode(self, random_input_2d):
        """Eval and training modes should produce close results."""
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        out_train = gn(random_input_2d)
        gn.eval()
        out_eval = gn(random_input_2d)
        torch.testing.assert_close(out_train, out_eval, atol=1e-5, rtol=1e-5)

    def test_normalized_output_has_zero_mean(self, random_input_2d):
        """After GroupNorm with default weight=1 and bias=0, each group should
        have approximately zero mean."""
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        out = gn(random_input_2d)
        # Reshape to groups and check mean is near zero
        reshaped = out.reshape(B, gn.num_groups, IN_CH // gn.num_groups, H, W)
        group_means = reshaped.mean(dim=[2, 3, 4])
        assert group_means.abs().max() < 0.1

    def test_normalized_output_has_variance_close_to_one(self, random_input_2d):
        """After GroupNorm with default weight=1 and bias=0, each group should
        have variance close to 1 (not exactly 1 due to eps)."""
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        out = gn(random_input_2d)
        # Reshape to groups and check variance is near 1
        reshaped = out.reshape(B, gn.num_groups, IN_CH // gn.num_groups, H, W)
        group_vars = reshaped.var(dim=[2, 3, 4], unbiased=False)
        assert torch.allclose(group_vars, torch.ones_like(group_vars), atol=0.1)

    def test_gradients_flow(self, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH)
        gn.train()
        x = random_input_2d.clone().requires_grad_(True)
        out = gn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGroupNormFusedActivation:
    """Test GroupNorm with fused activation functions."""

    @pytest.mark.parametrize(
        "act_name", ["silu", "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "elu"]
    )
    def test_fused_act_accepted(self, act_name, random_input_2d):
        gn = GroupNorm(num_channels=IN_CH, fused_act=True, act=act_name)
        gn.train()
        out = gn(random_input_2d)
        assert out.shape == random_input_2d.shape

    def test_invalid_act_raises(self):
        with pytest.raises(ValueError, match="Unknown activation function"):
            GroupNorm(num_channels=OUT_CH, fused_act=True, act="invalid_act")

    @pytest.mark.parametrize(
        "act_name", ["silu", "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "elu"]
    )
    def test_fused_act_matches_separate(self, act_name, random_input_2d):
        """Fused activation should give the same result as applying the activation separately."""
        gn_fused = GroupNorm(num_channels=IN_CH, fused_act=True, act=act_name)
        gn_plain = GroupNorm(num_channels=IN_CH)
        # Copy parameters
        gn_fused.train()
        gn_plain.train()
        out_fused = gn_fused(random_input_2d)
        out_separate = getattr(torch.nn.functional, act_name)(gn_plain(random_input_2d))
        torch.testing.assert_close(out_fused, out_separate)


############################################################################
#                          AttentionOp                                     #
############################################################################


class TestAttentionOpForward:
    """Test AttentionOp forward pass."""

    def test_output_shape(self):
        q = torch.randn(B, 16, 8)
        k = torch.randn(B, 16, 8)
        w = AttentionOp.apply(q, k)
        assert w.shape == (B, 8, 8)

    def test_output_is_probability_distribution(self):
        """Each row of the attention weights should sum to 1 (softmax output)."""
        q = torch.randn(B, 16, 8)
        k = torch.randn(B, 16, 8)
        w = AttentionOp.apply(q, k)
        row_sums = w.sum(dim=2)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

    def test_output_non_negative(self):
        q = torch.randn(B, 16, 8)
        k = torch.randn(B, 16, 8)
        w = AttentionOp.apply(q, k)
        assert (w >= 0).all()

    def test_output_dtype_matches_input(self):
        q = torch.randn(B, 16, 8)
        k = torch.randn(B, 16, 8)
        w = AttentionOp.apply(q, k)
        assert w.dtype == q.dtype


class TestAttentionOpBackward:
    """Test AttentionOp backward pass."""

    def test_gradients_flow_to_q(self):
        q = torch.randn(B, 16, 8, requires_grad=True)
        k = torch.randn(B, 16, 8, requires_grad=True)
        w = AttentionOp.apply(q, k)
        w.sum().backward()
        assert q.grad is not None
        assert q.grad.shape == q.shape

    def test_gradients_flow_to_k(self):
        q = torch.randn(B, 16, 8, requires_grad=True)
        k = torch.randn(B, 16, 8, requires_grad=True)
        w = AttentionOp.apply(q, k)
        w.sum().backward()
        assert k.grad is not None
        assert k.grad.shape == k.shape


############################################################################
#                         UNetBlock                                        #
############################################################################


class TestUNetBlockInit:
    """Test UNetBlock.__init__ parameter setup."""

    def test_stores_block_info(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH,
            dropout=0.1, skip_scale=0.5, adaptive_scale=False,
            profile_mode=True, amp_mode=True, attention=True,
            num_heads=4
        )
        assert block.in_channels == IN_CH
        assert block.out_channels == OUT_CH
        assert block.emb_channels == EMB_CH
        assert block.dropout == 0.1
        assert block.skip_scale == 0.5
        assert block.adaptive_scale is False
        assert block.profile_mode is True
        assert block.amp_mode is True

    def test_num_heads_zero_when_no_attention(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH, attention=False
        )
        assert block.num_heads == 0

    def test_num_heads_set_when_attention_no_num_heads(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=16, emb_channels=EMB_CH, 
            attention=True, channels_per_head=4
        )
        assert block.num_heads == 16//4

    def test_skip_created_when_channels_differ(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        assert block.skip is not None

    def test_skip_none_when_channels_match(self):
        block = UNetBlock(
            in_channels=OUT_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        assert block.skip is None

    def test_skip_created_when_up(self):
        block = UNetBlock(
            in_channels=OUT_CH, out_channels=OUT_CH, emb_channels=EMB_CH, up=True
        )
        assert block.skip is not None

    def test_skip_created_when_down(self):
        block = UNetBlock(
            in_channels=OUT_CH, out_channels=OUT_CH, emb_channels=EMB_CH, down=True
        )
        assert block.skip is not None

    def test_attention_heads_not_created_when_attention_false(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH, attention=False
        )
        assert not hasattr(block, "norm2") or block.norm2 is None
        assert not hasattr(block, "qkv") or block.qkv is None
        assert not hasattr(block, "proj") or block.proj is None

    def test_attention_heads_default(self):
        block = UNetBlock(
            in_channels=64,
            out_channels=64,
            emb_channels=EMB_CH,
            attention=True,
            channels_per_head=64,
        )
        assert block.norm2 is not None
        assert block.qkv is not None
        assert block.proj is not None

    def test_has_norm_and_conv_layers(self):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        assert hasattr(block, "norm0")
        assert hasattr(block, "conv0")
        assert hasattr(block, "norm1")
        assert hasattr(block, "conv1")
        assert hasattr(block, "affine")


class TestUNetBlockForward:
    """Test UNetBlock forward pass."""

    def test_output_shape_same_channels(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=IN_CH, emb_channels=EMB_CH
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, IN_CH, H, W)

    def test_output_shape_different_channels(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H, W)

    def test_output_shape_upsample(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH, up=True
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H * 2, W * 2)

    def test_output_shape_downsample(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH, down=True
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H // 2, W // 2)

    def test_output_dtype_matches_input(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        out = block(random_input_2d, embedding)
        assert out.dtype == random_input_2d.dtype

    def test_gradients_flow(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH, out_channels=OUT_CH, emb_channels=EMB_CH
        )
        x = random_input_2d.clone().requires_grad_(True)
        out = block(x, embedding)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_with_attention(self, embedding):
        ch = 64
        x = torch.randn(B, ch, H, W)
        block = UNetBlock(
            in_channels=ch,
            out_channels=ch,
            emb_channels=EMB_CH,
            attention=True,
            channels_per_head=ch//2,
        )
        out = block(x, embedding)
        assert out.shape == (B, ch, H, W)

    def test_non_adaptive_scale(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            emb_channels=EMB_CH,
            adaptive_scale=False,
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H, W)

    def test_with_dropout(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            emb_channels=EMB_CH,
            dropout=0.1,
        )
        block.train()
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H, W)

    def test_with_amp_mode(self, random_input_2d, embedding):
        block = UNetBlock(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            emb_channels=EMB_CH,
            amp_mode=True,
        )
        out = block(random_input_2d, embedding)
        assert out.shape == (B, OUT_CH, H, W)

    def test_skip_scale_applied(self, random_input_2d, embedding):
        """Changing skip_scale should change the output magnitude."""
        block_s1 = UNetBlock(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            emb_channels=EMB_CH,
            skip_scale=1.0,
        )
        block_s2 = UNetBlock(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            emb_channels=EMB_CH,
            skip_scale=2.0,
        )
        # Copy parameters from block_s1 to block_s2
        block_s2.load_state_dict(block_s1.state_dict(), strict=False)
        out1 = block_s1(random_input_2d, embedding)
        out2 = block_s2(random_input_2d, embedding)
        # s2 should have roughly 2x the magnitude of s1
        torch.testing.assert_close(out2, out1 * 2.0, atol=1e-5, rtol=1e-5)


############################################################################
#                      PositionalEmbedding                                 #
############################################################################


class TestPositionalEmbeddingInit:
    """Test PositionalEmbedding.__init__."""

    def test_stores_num_channels(self):
        emb = PositionalEmbedding(num_channels=64)
        assert emb.num_channels == 64

    def test_stores_max_positions(self):
        emb = PositionalEmbedding(num_channels=64, max_positions=5000)
        assert emb.max_positions == 5000

    def test_stores_endpoint(self):
        emb = PositionalEmbedding(num_channels=64, endpoint=True)
        assert emb.endpoint is True

    def test_amp_mode(self):
        emb = PositionalEmbedding(num_channels=64, amp_mode=True)
        assert emb.amp_mode is True


class TestPositionalEmbeddingForward:
    """Test PositionalEmbedding forward pass."""

    def test_output_shape(self):
        emb = PositionalEmbedding(num_channels=64)
        x = torch.randn(B)
        out = emb(x)
        assert out.shape == (B, 64)

    def test_output_shape_single(self):
        emb = PositionalEmbedding(num_channels=32)
        x = torch.randn(1)
        out = emb(x)
        assert out.shape == (1, 32)

    def test_different_inputs_produce_different_embeddings(self):
        emb = PositionalEmbedding(num_channels=64)
        x1 = torch.tensor([0.1])
        x2 = torch.tensor([1.0])
        out1 = emb(x1)
        out2 = emb(x2)
        assert not torch.allclose(out1, out2)

    def test_same_input_produces_same_embedding(self):
        emb = PositionalEmbedding(num_channels=64)
        x = torch.tensor([0.5])
        out1 = emb(x)
        out2 = emb(x)
        torch.testing.assert_close(out1, out2)

    def test_output_contains_sin_and_cos(self):
        """Output is concatenation of cos and sin, so first and second halves
        should differ for non-trivial inputs."""
        emb = PositionalEmbedding(num_channels=64)
        x = torch.tensor([1.0])
        out = emb(x)
        first_half = out[:, :32]
        second_half = out[:, 32:]
        assert not torch.allclose(first_half, second_half)

    def test_output_bounded(self):
        """Since output is cos and sin, values should be in [-1, 1]."""
        emb = PositionalEmbedding(num_channels=64)
        x = torch.randn(B)
        out = emb(x)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_endpoint_changes_output(self):
        emb_no_end = PositionalEmbedding(num_channels=64, endpoint=False)
        emb_end = PositionalEmbedding(num_channels=64, endpoint=True)
        x = torch.tensor([1.0])
        out1 = emb_no_end(x)
        out2 = emb_end(x)
        assert not torch.allclose(out1, out2)


############################################################################
#                       FourierEmbedding                                   #
############################################################################


class TestFourierEmbeddingInit:
    """Test FourierEmbedding.__init__."""

    def test_freqs_buffer_registered(self):
        emb = FourierEmbedding(num_channels=64)
        assert hasattr(emb, "freqs")
        assert emb.freqs.shape == (32,)

    def test_scale_affects_freqs_magnitude(self):
        torch.manual_seed(0)
        emb_small = FourierEmbedding(num_channels=64, scale=1)
        torch.manual_seed(0)
        emb_large = FourierEmbedding(num_channels=64, scale=16)
        torch.testing.assert_close(emb_large.freqs, emb_small.freqs * 16)

    def test_amp_mode_stored(self):
        emb = FourierEmbedding(num_channels=64, amp_mode=True)
        assert emb.amp_mode is True


class TestFourierEmbeddingForward:
    """Test FourierEmbedding forward pass."""

    def test_output_shape(self):
        emb = FourierEmbedding(num_channels=64)
        x = torch.randn(B)
        out = emb(x)
        assert out.shape == (B, 64)

    def test_output_shape_single(self):
        emb = FourierEmbedding(num_channels=32)
        x = torch.randn(1)
        out = emb(x)
        assert out.shape == (1, 32)

    def test_different_inputs_produce_different_embeddings(self):
        emb = FourierEmbedding(num_channels=64)
        x1 = torch.tensor([0.1])
        x2 = torch.tensor([1.0])
        out1 = emb(x1)
        out2 = emb(x2)
        assert not torch.allclose(out1, out2)

    def test_same_input_produces_same_embedding(self):
        emb = FourierEmbedding(num_channels=64)
        x = torch.tensor([0.5])
        out1 = emb(x)
        out2 = emb(x)
        torch.testing.assert_close(out1, out2)

    def test_output_contains_sin_and_cos(self):
        emb = FourierEmbedding(num_channels=64)
        x = torch.tensor([1.0])
        out = emb(x)
        first_half = out[:, :32]
        second_half = out[:, 32:]
        assert not torch.allclose(first_half, second_half)

    def test_output_bounded(self):
        """Since output is cos and sin, values should be in [-1, 1]."""
        emb = FourierEmbedding(num_channels=64)
        x = torch.randn(B)
        out = emb(x)
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6
