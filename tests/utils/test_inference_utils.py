import pytest
import os
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from hirad.utils.inference_utils import (
    calculate_bounds,
    regression_step,
    diffusion_step,
    save_results_as_torch,
)


############################################################################
#                          calculate_bounds                                 #
############################################################################


class TestCalculateBounds:
    """Tests for calculate_bounds."""

    def test_single_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vmin, vmax = calculate_bounds(arr)
        assert vmin == 1.0
        assert vmax == 5.0

    def test_multiple_arrays(self):
        a = np.array([1.0, 5.0])
        b = np.array([-3.0, 2.0])
        c = np.array([0.0, 10.0])
        vmin, vmax = calculate_bounds(a, b, c)
        assert vmin == -3.0
        assert vmax == 10.0

    def test_no_arrays(self):
        vmin, vmax = calculate_bounds()
        assert vmin is None
        assert vmax is None

    def test_all_none(self):
        vmin, vmax = calculate_bounds(None, None)
        assert vmin is None
        assert vmax is None

    def test_some_none(self):
        arr = np.array([2.0, 8.0])
        vmin, vmax = calculate_bounds(None, arr, None)
        assert vmin == 2.0
        assert vmax == 8.0

    def test_single_value_array(self):
        arr = np.array([42.0])
        vmin, vmax = calculate_bounds(arr)
        assert vmin == 42.0
        assert vmax == 42.0

    def test_negative_values(self):
        arr = np.array([-10.0, -5.0, -1.0])
        vmin, vmax = calculate_bounds(arr)
        assert vmin == -10.0
        assert vmax == -1.0

    def test_masked_array(self):
        data = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        masked = np.ma.masked_invalid(data)
        vmin, vmax = calculate_bounds(masked)
        assert vmin == 1.0
        assert vmax == 4.0

    def test_masked_array_all_masked(self):
        data = np.ma.array([1.0, 2.0], mask=[True, True])
        vmin, vmax = calculate_bounds(data)
        assert vmin == None
        assert vmax == None

    def test_mixed_masked_and_regular(self):
        regular = np.array([0.0, 5.0])
        masked = np.ma.masked_invalid(np.array([np.nan, 10.0, np.nan]))
        vmin, vmax = calculate_bounds(regular, masked)
        assert vmin == 0.0
        assert vmax == 10.0

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        vmin, vmax = calculate_bounds(arr)
        assert vmin == 1.0
        assert vmax == 4.0

    def test_scalar_input(self):
        vmin, vmax = calculate_bounds(np.float64(3.14))
        assert vmin == pytest.approx(3.14)
        assert vmax == pytest.approx(3.14)


############################################################################
#                          regression_step                                 #
############################################################################


class TestRegressionStep:
    """Tests for regression_step."""

    def _make_mock_net(self, output_shape):
        """Create a mock network that returns a tensor of the given shape."""
        net = MagicMock()
        net.return_value = torch.randn(output_shape)
        return net

    def test_batch_size_greater_than_1_raises(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(2, 3, 8, 8)  # batch_size=2
        latents_shape = torch.Size([1, 4, 8, 8])
        with pytest.raises(ValueError, match="batch size of 1"):
            regression_step(net, img_lr, latents_shape)

    def test_batch_size_1_succeeds(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        latents_shape = torch.Size([1, 4, 8, 8])
        result = regression_step(net, img_lr, latents_shape)
        assert result.shape == torch.Size([1, 4, 8, 8])

    def test_output_replicated_when_latents_batch_gt_1(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        latents_shape = torch.Size([5, 4, 8, 8])
        result = regression_step(net, img_lr, latents_shape)
        assert result.shape == torch.Size([5, 4, 8, 8])

    def test_net_called_with_img_lr(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        latents_shape = torch.Size([1, 4, 8, 8])
        regression_step(net, img_lr, latents_shape)
        assert net.called

    def test_with_lead_time_label(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        latents_shape = torch.Size([1, 4, 8, 8])
        lead_time = torch.tensor([1.0])
        result = regression_step(
            net, img_lr, latents_shape, lead_time_label=lead_time
        )
        assert result.shape == torch.Size([1, 4, 8, 8])
        # Verify lead_time_label was passed to the net
        _, kwargs = net.call_args
        assert "lead_time_label" in kwargs

    def test_with_static_channels(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        static = torch.randn(1, 2, 8, 8)
        latents_shape = torch.Size([1, 4, 8, 8])
        result = regression_step(
            net, img_lr, latents_shape, static_channels=static
        )
        assert result.shape == torch.Size([1, 4, 8, 8])
        # Net should receive img_lr concatenated with static channels (3+2=5)
        _, kwargs = net.call_args
        assert kwargs["img_lr"].shape[1] == 5

    def test_with_date_embedding(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        date_emb = torch.randn(1, 4)
        latents_shape = torch.Size([1, 4, 8, 8])
        result = regression_step(
            net, img_lr, latents_shape, date_embedding=date_emb
        )
        assert result.shape == torch.Size([1, 4, 8, 8])
        # Net should receive img_lr concatenated with date embedding (3+4=7)
        _, kwargs = net.call_args
        assert kwargs["img_lr"].shape[1] == 7

    def test_with_all_optional_inputs(self):
        net = self._make_mock_net((1, 4, 8, 8))
        img_lr = torch.randn(1, 3, 8, 8)
        static = torch.randn(1, 2, 8, 8)
        date_emb = torch.randn(1, 4)
        lead_time = torch.tensor([1.0])
        latents_shape = torch.Size([1, 4, 8, 8])
        result = regression_step(
            net,
            img_lr,
            latents_shape,
            lead_time_label=lead_time,
            static_channels=static,
            date_embedding=date_emb,
        )
        assert result.shape == torch.Size([1, 4, 8, 8])
        # img_lr should have 3 + 2 (static) + 4 (date) = 9 channels
        _, kwargs = net.call_args
        assert kwargs["img_lr"].shape[1] == 9


############################################################################
#                          diffusion_step                                  #
############################################################################


class TestDiffusionStep:
    """Tests for diffusion_step."""

    def test_img_lr_shape_mismatch_raises(self):
        net = MagicMock()
        sampler_fn = MagicMock()
        img_lr = torch.randn(1, 3, 16, 16)
        with pytest.raises(ValueError, match="does not match expected shape"):
            diffusion_step(
                net=net,
                sampler_fn=sampler_fn,
                img_shape=(32, 32),
                img_out_channels=4,
                rank_batches=[[0]],
                img_lr=img_lr,
                rank=0,
                device=torch.device("cpu"),
            )

    def test_mean_hr_shape_mismatch_raises(self):
        net = MagicMock()
        sampler_fn = MagicMock()
        img_lr = torch.randn(1, 3, 32, 32)
        mean_hr = torch.randn(1, 4, 16, 16)
        with pytest.raises(ValueError, match="does not match expected shape"):
            diffusion_step(
                net=net,
                sampler_fn=sampler_fn,
                img_shape=(32, 32),
                img_out_channels=4,
                rank_batches=[[0]],
                img_lr=img_lr,
                rank=0,
                device=torch.device("cpu"),
                mean_hr=mean_hr,
            )

    def test_mean_hr_batch_size_not_1_raises(self):
        net = MagicMock()
        sampler_fn = MagicMock()
        img_lr = torch.randn(1, 3, 32, 32)
        mean_hr = torch.randn(2, 4, 32, 32)
        with pytest.raises(ValueError, match="batch size 1"):
            diffusion_step(
                net=net,
                sampler_fn=sampler_fn,
                img_shape=(32, 32),
                img_out_channels=4,
                rank_batches=[[0]],
                img_lr=img_lr,
                rank=0,
                device=torch.device("cpu"),
                mean_hr=mean_hr,
            )

    def test_empty_rank_batches(self):
        net = MagicMock()
        sampler_fn = MagicMock()
        img_lr = torch.randn(1, 3, 8, 8)
        # Empty batches of seeds
        with pytest.raises(ValueError, match="rank_batches is empty"):
            diffusion_step(
                net=net,
                sampler_fn=sampler_fn,
                img_shape=(8, 8),
                img_out_channels=4,
                rank_batches=[],
                img_lr=img_lr,
                rank=0,
                device=torch.device("cpu"),
            )

    def test_missmatch_batch_size_and_image_shape(self):
        net = MagicMock()
        sampler_fn = MagicMock()
        img_lr = torch.randn(1, 3, 8, 8)
        # rank_batches has batch size 2 but img_shape is for batch size 1
        with pytest.raises(ValueError, match="does not match img_lr batch size"):
            diffusion_step(
                net=net,
                sampler_fn=sampler_fn,
                img_shape=(8, 8),
                img_out_channels=4,
                rank_batches=[[0,1], [2,3]],
                img_lr=img_lr,
                rank=0,
                device=torch.device("cpu"),
            )

    def test_generates_correct_number_of_samples(self):
        net = MagicMock()
        generated = torch.randn(1, 4, 8, 8)
        sampler_fn = MagicMock(return_value=generated)
        img_lr = torch.randn(1, 3, 8, 8)
        result = diffusion_step(
            net=net,
            sampler_fn=sampler_fn,
            img_shape=(8, 8),
            img_out_channels=4,
            rank_batches=[[0], [1], [2]],
            img_lr=img_lr,
            rank=0,
            device=torch.device("cpu"),
        )
        assert result.shape == torch.Size([3, 4, 8, 8])
        assert sampler_fn.call_count == 3

    def test_passes_additional_args_to_sampler(self):
        net = MagicMock()
        generated = torch.randn(1, 4, 8, 8)
        sampler_fn = MagicMock(return_value=generated)
        img_lr = torch.randn(1, 3, 8, 8)
        mean_hr = torch.randn(1, 4, 8, 8)
        lead_time = torch.tensor([1.0])
        static = torch.randn(1, 2, 8, 8)
        date_emb = torch.randn(1, 4)

        diffusion_step(
            net=net,
            sampler_fn=sampler_fn,
            img_shape=(8, 8),
            img_out_channels=4,
            rank_batches=[[42]],
            img_lr=img_lr,
            rank=0,
            device=torch.device("cpu"),
            mean_hr=mean_hr,
            lead_time_label=lead_time,
            static_channels=static,
            date_embedding=date_emb,
        )

        _, kwargs = sampler_fn.call_args
        assert "mean_hr" in kwargs
        assert "lead_time_label" in kwargs
        assert "static_channels" in kwargs
        assert "date_embedding" in kwargs


############################################################################
#                       save_results_as_torch                              #
############################################################################


class TestSaveResultsAsTorch:
    """Tests for save_results_as_torch."""

    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "results" / "nested"
        image_pred = torch.randn(2, 4, 8, 8)
        image_hr = torch.randn(1, 4, 8, 8)
        image_lr = torch.randn(1, 3, 8, 8)
        mean_pred = torch.randn(1, 4, 8, 8)

        save_results_as_torch(
            str(output_dir), "step_0", image_pred, image_hr, image_lr, mean_pred
        )
        assert output_dir.exists()

    def test_saves_all_files_with_mean(self, tmp_path):
        image_pred = torch.randn(2, 4, 8, 8)
        image_hr = torch.randn(1, 4, 8, 8)
        image_lr = torch.randn(1, 3, 8, 8)
        mean_pred = torch.randn(1, 4, 8, 8)

        save_results_as_torch(
            str(tmp_path), "step_0", image_pred, image_hr, image_lr, mean_pred
        )

        assert os.path.isfile(tmp_path / "step_0-regression-prediction")
        assert os.path.isfile(tmp_path / "step_0-target")
        assert os.path.isfile(tmp_path / "step_0-predictions")
        assert os.path.isfile(tmp_path / "step_0-baseline")

    def test_skips_regression_when_mean_is_none(self, tmp_path):
        image_pred = torch.randn(2, 4, 8, 8)
        image_hr = torch.randn(1, 4, 8, 8)
        image_lr = torch.randn(1, 3, 8, 8)

        save_results_as_torch(
            str(tmp_path), "step_1", image_pred, image_hr, image_lr, None
        )

        assert not os.path.isfile(tmp_path / "step_1-regression-prediction")
        assert os.path.isfile(tmp_path / "step_1-target")
        assert os.path.isfile(tmp_path / "step_1-predictions")
        assert os.path.isfile(tmp_path / "step_1-baseline")

    def test_saved_tensors_are_loadable_and_correct(self, tmp_path):
        image_pred = torch.randn(2, 4, 8, 8)
        image_hr = torch.randn(1, 4, 8, 8)
        image_lr = torch.randn(1, 3, 8, 8)
        mean_pred = torch.randn(1, 4, 8, 8)

        save_results_as_torch(
            str(tmp_path), "t0", image_pred, image_hr, image_lr, mean_pred
        )

        loaded_pred = torch.load(tmp_path / "t0-predictions", weights_only=True)
        loaded_hr = torch.load(tmp_path / "t0-target", weights_only=True)
        loaded_lr = torch.load(tmp_path / "t0-baseline", weights_only=True)
        loaded_mean = torch.load(tmp_path / "t0-regression-prediction", weights_only=True)

        assert torch.equal(loaded_pred, image_pred)
        assert torch.equal(loaded_hr, image_hr)
        assert torch.equal(loaded_lr, image_lr)
        assert torch.equal(loaded_mean, mean_pred)

    def test_different_time_steps_dont_overwrite(self, tmp_path):
        t1 = torch.randn(1, 4, 8, 8)
        t2 = torch.randn(1, 4, 8, 8)

        save_results_as_torch(str(tmp_path), "step_0", t1, t1, t1, None)
        save_results_as_torch(str(tmp_path), "step_1", t2, t2, t2, None)

        loaded_0 = torch.load(tmp_path / "step_0-target", weights_only=True)
        loaded_1 = torch.load(tmp_path / "step_1-target", weights_only=True)

        assert torch.equal(loaded_0, t1)
        assert torch.equal(loaded_1, t2)