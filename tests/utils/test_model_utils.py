import pytest
import numpy as np
import torch

from hirad.utils.model_utils import weight_init


class TestWeightInitShape:
    """Test that weight_init returns tensors of the correct shape."""

    @pytest.mark.parametrize("mode", [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
    ])
    @pytest.mark.parametrize("shape", [
        (1,),
        (3, 3),
        (64, 32, 3, 3),
        (128, 64, 5, 5),
    ])
    def test_output_shape(self, mode, shape):
        result = weight_init(shape, mode, fan_in=32, fan_out=64)
        assert result.shape == torch.Size(shape)

    @pytest.mark.parametrize("mode", [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
    ])
    def test_output_is_tensor(self, mode):
        result = weight_init((4, 4), mode, fan_in=4, fan_out=4)
        assert isinstance(result, torch.Tensor)


class TestWeightInitValues:
    """Test that weight_init produces values within expected bounds."""

    def test_xavier_uniform_bound(self):
        fan_in, fan_out = 256, 512
        bound = np.sqrt(6 / (fan_in + fan_out))
        result = weight_init((10000,), "xavier_uniform", fan_in, fan_out)
        assert result.min() >= -bound - 1e-7
        assert result.max() <= bound + 1e-7

    def test_kaiming_uniform_bound(self):
        fan_in, fan_out = 256, 512
        bound = np.sqrt(3 / fan_in)
        result = weight_init((10000,), "kaiming_uniform", fan_in, fan_out)
        assert result.min() >= -bound - 1e-7
        assert result.max() <= bound + 1e-7

    def test_xavier_normal_mean_and_std(self):
        fan_in, fan_out = 256, 512
        expected_std = np.sqrt(2 / (fan_in + fan_out))
        result = weight_init((100000,), "xavier_normal", fan_in, fan_out)
        assert abs(result.mean().item()) < 0.05
        assert abs(result.std().item() - expected_std) < 0.01

    def test_kaiming_normal_mean_and_std(self):
        fan_in, fan_out = 256, 512
        expected_std = np.sqrt(1 / fan_in)
        result = weight_init((100000,), "kaiming_normal", fan_in, fan_out)
        assert abs(result.mean().item()) < 0.05
        assert abs(result.std().item() - expected_std) < 0.01


class TestWeightInitSymmetry:
    """Test that uniform modes are centered around zero."""

    @pytest.mark.parametrize("mode", ["xavier_uniform", "kaiming_uniform"])
    def test_uniform_centered_around_zero(self, mode):
        result = weight_init((100000,), mode, fan_in=128, fan_out=128)
        assert abs(result.mean().item()) < 0.05


class TestWeightInitScaling:
    """Test that changing fan_in/fan_out changes the scale of outputs."""

    @pytest.mark.parametrize("mode", [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
    ])
    def test_larger_fan_in_reduces_scale(self, mode):
        small_fan = weight_init((10000,), mode, fan_in=16, fan_out=16)
        large_fan = weight_init((10000,), mode, fan_in=1024, fan_out=1024)
        assert small_fan.std() > large_fan.std()


class TestWeightInitInvalidMode:
    """Test that invalid modes raise ValueError."""

    @pytest.mark.parametrize("mode", [
        "invalid",
        "",
        "Xavier_uniform",
        "KAIMING_NORMAL",
        "he_normal",
        "glorot_uniform",
    ])
    def test_invalid_mode_raises_value_error(self, mode):
        with pytest.raises(ValueError, match="Invalid init mode"):
            weight_init((4, 4), mode, fan_in=4, fan_out=4)