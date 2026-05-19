# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from hirad.training.training_manager import TrainingManagerBase, TrainingManagerCorrDiff


# ---------------------------------------------------------------------------
#  Helpers / fixtures
# ---------------------------------------------------------------------------

B, C_IN, C_OUT, C_STATIC, H, W = 2, 4, 3, 2, 64, 64


def _make_mock_dist(device="cpu", rank=0, world_size=1):
    """Return a lightweight mock of DistributedManager."""
    dist = MagicMock()
    dist.device = torch.device(device)
    dist.rank = rank
    dist.world_size = world_size
    dist.local_rank = 0
    return dist


def _make_mock_dataset(
    n_input=C_IN,
    n_output=C_OUT,
    n_static=C_STATIC,
    img_shape=(H, W),
    static_data=None,
    trim_edge=0,
):
    """Return a MagicMock that satisfies the DownscalingDataset interface."""
    ds = MagicMock()
    ds.input_channels.return_value = [MagicMock()] * n_input
    ds.output_channels.return_value = [MagicMock()] * n_output
    ds.static_channels.return_value = [MagicMock()] * n_static
    ds.image_shape.return_value = img_shape
    ds.get_static_data.return_value = static_data
    ds.trim_edge = trim_edge
    # normalize / denormalize are identity by default
    ds.normalize_input.side_effect = lambda x: x
    ds.normalize_output.side_effect = lambda x: x
    ds.denormalize_input.side_effect = lambda x: x
    ds.denormalize_output.side_effect = lambda x: x
    # interpolator returns input reshaped (identity)
    ds.interpolator.side_effect = lambda x: x
    # make_time_grids returns a dummy tensor
    ds.make_time_grids.return_value = torch.zeros(B, 8)
    ds.regrid_indices_real = None
    ds.regrid_weights_real = None
    return ds


def _make_manager_corrdiff(
    dist=None,
    dataset=None,
    input_dtype=torch.float32,
    img_shape=(H, W),
    n_month_hour_channels=0,
    fp16=False,
    enable_amp=False,
    amp_dtype=torch.bfloat16,
    use_apex_gn=False,
    is_real_target=False,
    songunet_checkpoint_level=0,
    use_patching=False,
    hr_mean_conditioning=False,
    profile_mode=False,
    logging_method=None,
):
    """Convenience factory for TrainingManagerCorrDiff with sensible defaults."""
    if dist is None:
        dist = _make_mock_dist()
    if dataset is None:
        dataset = _make_mock_dataset()
    return TrainingManagerCorrDiff(
        dist=dist,
        logger=MagicMock(),
        dataset=dataset,
        input_dtype=input_dtype,
        img_shape=img_shape,
        n_month_hour_channels=n_month_hour_channels,
        fp16=fp16,
        enable_amp=enable_amp,
        amp_dtype=amp_dtype,
        use_apex_gn=use_apex_gn,
        is_real_target=is_real_target,
        songunet_checkpoint_level=songunet_checkpoint_level,
        use_patching=use_patching,
        hr_mean_conditioning=hr_mean_conditioning,
        profile_mode=profile_mode,
        logging_method=logging_method,
    )


############################################################################
#                     TrainingManagerBase (abstract)                        #
############################################################################


class TestTrainingManagerBase:
    """Test the abstract base class contract."""

    def test_cannot_instantiate_directly(self):
        """ABC should not be instantiable without implementing abstract methods."""
        with pytest.raises(TypeError):
            TrainingManagerBase(
                dist=_make_mock_dist(), logger=MagicMock()
            )

    def test_concrete_subclass_must_implement_all(self):
        """A subclass missing an abstract method should fail to instantiate."""

        class Incomplete(TrainingManagerBase):
            def load_and_preprocess_batch(self):
                pass

            def get_static_data(self):
                pass

            def create_model(self):
                pass
            # run_validation is missing

        with pytest.raises(TypeError):
            Incomplete(dist=_make_mock_dist(), logger=MagicMock())

    def test_stores_dist_and_logger(self):
        """Concrete subclass should inherit dist/logger attributes."""
        mgr = _make_manager_corrdiff()
        assert mgr.dist is not None
        assert mgr.logger is not None


############################################################################
#              TrainingManagerCorrDiff — __init__                           #
############################################################################


class TestCorrDiffInit:
    """Test that __init__ stores all configuration values."""

    def test_stores_dataset(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        assert mgr.dataset is ds

    def test_stores_img_shape(self):
        mgr = _make_manager_corrdiff(img_shape=(128, 256))
        assert mgr.img_shape == (128, 256)

    def test_stores_precision_flags(self):
        mgr = _make_manager_corrdiff(input_dtype=torch.float32, fp16=True, enable_amp=True, amp_dtype=torch.float16)
        assert mgr.fp16 is True
        assert mgr.enable_amp is True
        assert mgr.amp_dtype is torch.float16
        assert mgr.input_dtype is torch.float32

    def test_stores_apex_gn_flag(self):
        mgr = _make_manager_corrdiff(use_apex_gn=True)
        assert mgr.use_apex_gn is True

    def test_stores_profile_mode_flag(self):
        mgr = _make_manager_corrdiff(profile_mode=True)
        assert mgr.profile_mode is True

    def test_stores_is_real_target(self):
        mgr = _make_manager_corrdiff(is_real_target=True)
        assert mgr.is_real_target is True

    def test_stores_patching_and_hr_mean_conditioning_flags(self):
        mgr = _make_manager_corrdiff(use_patching=True, hr_mean_conditioning=True)
        assert mgr.use_patching is True
        assert mgr.hr_mean_conditioning is True

    def test_stores_logging_method(self):
        mgr = _make_manager_corrdiff(logging_method="mlflow")
        assert mgr.logging_method == "mlflow"

    def test_stores_n_month_hour_channels(self):
        mgr = _make_manager_corrdiff(n_month_hour_channels=6)
        assert mgr.n_month_hour_channels == 6

    def test_stores_songunet_checkpoint_level(self):
        mgr = _make_manager_corrdiff(songunet_checkpoint_level=2)
        assert mgr.songunet_checkpoint_level == 2



############################################################################
#          TrainingManagerCorrDiff — get_static_data                        #
############################################################################


class TestGetStaticData:
    """Tests for TrainingManagerCorrDiff.get_static_data."""

    def test_returns_none_when_dataset_has_no_static(self):
        ds = _make_mock_dataset(static_data=None)
        mgr = _make_manager_corrdiff(dataset=ds)
        assert mgr.get_static_data() is None

    def test_returns_tensor_from_numpy(self):
        """numpy static data should be converted to a torch tensor."""
        static_np = np.random.randn(C_STATIC, H, W).astype(np.float32)
        ds = _make_mock_dataset(static_data=static_np)
        mgr = _make_manager_corrdiff(dataset=ds)
        result = mgr.get_static_data()
        assert isinstance(result, torch.Tensor)

    def test_returns_tensor_from_tensor(self):
        """torch tensor static data should also be handled."""
        static_t = torch.randn(C_STATIC, H, W)
        ds = _make_mock_dataset(static_data=static_t)
        mgr = _make_manager_corrdiff(dataset=ds)
        result = mgr.get_static_data()
        assert isinstance(result, torch.Tensor)

    def test_adds_batch_dim(self):
        """Result should have a leading batch dim of 1."""
        static_np = np.random.randn(C_STATIC, H, W).astype(np.float32)
        ds = _make_mock_dataset(static_data=static_np)
        mgr = _make_manager_corrdiff(dataset=ds)
        result = mgr.get_static_data()
        assert result.shape[0] == 1

    def test_flips_height(self):
        """Static data should be flipped along the last-2 (height) dim."""
        static_np = np.arange(H).reshape(1, H, 1).repeat(W, axis=2).astype(np.float32)
        ds = _make_mock_dataset(n_static=1, static_data=static_np)
        mgr = _make_manager_corrdiff(dataset=ds)
        result = mgr.get_static_data()
        # After flip(-2): first row should be the last row of the original
        expected_first_row = float(H - 1)
        assert result[0, 0, 0, 0].item() == pytest.approx(expected_first_row)

    def test_channels_last_when_apex_gn(self):
        """With use_apex_gn=True, output should use channels_last memory format."""
        static_np = np.random.randn(C_STATIC, H, W).astype(np.float32)
        ds = _make_mock_dataset(static_data=static_np)
        mgr = _make_manager_corrdiff(dataset=ds, use_apex_gn=True)
        result = mgr.get_static_data()
        assert result.is_contiguous(memory_format=torch.channels_last)

    def test_contiguous_when_no_apex_gn(self):
        """Without apex_gn, output should be standard contiguous."""
        static_np = np.random.randn(C_STATIC, H, W).astype(np.float32)
        ds = _make_mock_dataset(static_data=static_np)
        mgr = _make_manager_corrdiff(dataset=ds, use_apex_gn=False)
        result = mgr.get_static_data()
        assert result.is_contiguous()


############################################################################
#       TrainingManagerCorrDiff — load_and_preprocess_batch                 #
############################################################################


class TestLoadAndPreprocessBatch:
    """Tests for TrainingManagerCorrDiff.load_and_preprocess_batch."""

    @staticmethod
    def _make_iterator(img_clean, img_lr, date_str=None):
        """Wrap tensors into an iterator that yields a single batch."""
        batch = [img_clean, img_lr]
        if date_str is not None:
            batch.append(date_str)
        return iter([batch])

    def test_returns_three_elements(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        result = mgr.load_and_preprocess_batch(it)
        assert len(result) == 3  # img_clean, img_lr, date_embedding

    def test_date_embedding_is_none_when_no_month_hour(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, n_month_hour_channels=0)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        _, _, date_embedding = mgr.load_and_preprocess_batch(it)
        assert date_embedding is None

    def test_date_embedding_returned_when_month_hour(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, n_month_hour_channels=4)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W),
            torch.randn(B, C_IN, H * W),
            date_str="20240101-1800",
        )
        _, _, date_embedding = mgr.load_and_preprocess_batch(it)
        assert date_embedding is not None

    def test_imgs_flipped_and_reshaped(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        img_clean = torch.randn(B, C_OUT, H * W)
        img_lr = torch.randn(B, C_IN, H * W)
        it = self._make_iterator(img_clean, img_lr)
        img_clean_out, img_lr_out, _ = mgr.load_and_preprocess_batch(it)
        # Output should be flipped along height and reshaped to (B, C, H, W)
        assert img_clean_out.shape == (B, C_OUT, H, W)
        assert img_lr_out.shape == (B, C_IN, H, W)
        # Check that the first row of the output corresponds to the last row of the input after flip
        expected_first_row_clean = img_clean[:, :, -W:]
        expected_first_row_lr = img_lr[:, :, -W:]
        expected_last_row_clean = img_clean[:, :, :W]
        expected_last_row_lr = img_lr[:, :, :W]
        assert torch.allclose(img_clean_out[:, :, 0, :], expected_first_row_clean)
        assert torch.allclose(img_lr_out[:, :, 0, :], expected_first_row_lr)
        assert torch.allclose(img_clean_out[:, :, -1, :], expected_last_row_clean)
        assert torch.allclose(img_lr_out[:, :, -1, :], expected_last_row_lr)


    def test_img_clean_flipped_and_reshaped_when_real_target(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, is_real_target=True)
        img_clean = torch.randn(B, C_OUT, H * W)
        img_lr = torch.randn(B, C_IN, H * W)
        it = self._make_iterator(img_clean, img_lr)
        mock_regrid = MagicMock(side_effect=lambda x,y,z: x.reshape(*x.shape[:-1], *mgr.img_shape))
        with patch("hirad.training.training_manager.regrid_icon_to_rotlatlon", mock_regrid):
            img_clean_out, _, _ = mgr.load_and_preprocess_batch(it)
            # Output should be flipped along height and reshaped to (B, C, H, W)
            assert img_clean_out.shape == (B, C_OUT, H, W)
            expected_first_row_clean = img_clean[:, :, -W:]
            expected_last_row_clean = img_clean[:, :, :W]
            assert torch.allclose(img_clean_out[:, :, 0, :], expected_first_row_clean)
            assert torch.allclose(img_clean_out[:, :, -1, :], expected_last_row_clean)

    def test_img_clean_trimmed_when_trim_edge_positive(self):
        trim = 4
        ds = _make_mock_dataset(trim_edge=trim)
        mgr = _make_manager_corrdiff(dataset=ds, is_real_target=True)
        img_clean = torch.randn(B, C_OUT, (H + 2 * trim) * (W + 2 * trim))
        img_lr = torch.randn(B, C_IN, H * W )
        it = self._make_iterator(img_clean, img_lr)
        mock_regrid = MagicMock(side_effect=lambda x,y,z: x.reshape(*x.shape[:-1], *(H+2*trim, W+2*trim)))
        with patch("hirad.training.training_manager.regrid_icon_to_rotlatlon", mock_regrid):
            img_clean_out, _, _ = mgr.load_and_preprocess_batch(it)
            # Output should be trimmed by 'trim' pixels on each side, flipped, and reshaped to (B, C, H, W)
            assert img_clean_out.shape == (B, C_OUT, H, W)
            # expected_first_row_clean = img_clean[:, :, -(W + 2 * trim):- (W + 2 * trim) + W]
            # expected_last_row_clean = img_clean[:, :, trim:trim + W]
            expected_first_row_clean = img_clean[:, :, -((trim+1)*(W+2*trim))+trim:-((trim+1)*(W+2*trim))+trim+W]
            expected_last_row_clean = img_clean[:, :, trim*(W+2*trim+1):trim*(W+2*trim+1) + W]
            assert torch.allclose(img_clean_out[:, :, 0, :], expected_first_row_clean)
            assert torch.allclose(img_clean_out[:, :, -1, :], expected_last_row_clean)
    

    def test_calls_normalize_input(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        mgr.load_and_preprocess_batch(it)
        ds.normalize_input.assert_called_once()

    def test_calls_normalize_output(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        mgr.load_and_preprocess_batch(it)
        ds.normalize_output.assert_called_once()

    def test_calls_interpolator(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        mgr.load_and_preprocess_batch(it)
        ds.interpolator.assert_called_once()

    def test_real_target_calls_regrid_icon_to_latlon(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, is_real_target=True)
        mock_regrid = MagicMock(side_effect=lambda x,y,z: x.reshape(*x.shape[:-1], *mgr.img_shape))
        with patch("hirad.training.training_manager.regrid_icon_to_rotlatlon", mock_regrid):
            it = self._make_iterator(
                torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
            )
            mgr.load_and_preprocess_batch(it)
            mock_regrid.assert_called_once()

    def test_output_with_apex_gn_is_channels_last(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, use_apex_gn=True)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        img_clean, img_lr, _ = mgr.load_and_preprocess_batch(it)
        assert img_clean.is_contiguous(memory_format=torch.channels_last)
        assert img_lr.is_contiguous(memory_format=torch.channels_last)

    def test_output_without_apex_gn_is_contiguous(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, use_apex_gn=False)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        img_clean, img_lr, _ = mgr.load_and_preprocess_batch(it)
        assert img_clean.is_contiguous()
        assert img_lr.is_contiguous()

    def test_output_dtype_matches_input_dtype(self):
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        it = self._make_iterator(
            torch.randn(B, C_OUT, H * W), torch.randn(B, C_IN, H * W)
        )
        img_clean, img_lr, _ = mgr.load_and_preprocess_batch(it)
        assert img_clean.dtype == torch.float32
        assert img_lr.dtype == torch.float32


############################################################################
#            TrainingManagerCorrDiff — create_model                         #
############################################################################


class TestCreateModel:
    """Tests for TrainingManagerCorrDiff.create_model."""

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_diffusion_returns_edm(self, MockEDM):
        """'diffusion' model name should instantiate EDMPrecondSuperResolution."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 2}
        )
        MockEDM.assert_called_once()

    @patch("hirad.training.training_manager.UNet")
    def test_regression_returns_unet(self, MockUNet):
        """'regression' model name should instantiate UNet."""
        MockUNet.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "regression", {"N_grid_channels": 2}
        )
        MockUNet.assert_called_once()

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_patched_diffusion_returns_edm(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "patched_diffusion", {"N_grid_channels": 2}
        )
        MockEDM.assert_called_once()

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_lt_aware_patched_diffusion_returns_edm(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "lt_aware_patched_diffusion",
            {"N_grid_channels": 2, "lead_time_channels": 1},
        )
        MockEDM.assert_called_once()

    @patch("hirad.training.training_manager.UNet")
    def test_lt_aware_ce_regression_returns_unet(self, MockUNet):
        MockUNet.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "lt_aware_ce_regression",
            {"N_grid_channels": 2, "lead_time_channels": 1},
            )
        MockUNet.assert_called_once()

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_returns_model_and_args(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds)
        model, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 2}
        )
        assert model is not None
        assert isinstance(args, dict)

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_model_args_contain_resolution(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, img_shape=(32, 64))
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 2}
        )
        assert args["img_resolution"] == [32, 64]

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_model_args_contain_fp16_flag(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, fp16=True)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 2}
        )
        assert args["use_fp16"] is True

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_cfg_args_override_defaults(self, MockEDM):
        """cfg_model_args should override the default model_args."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, songunet_checkpoint_level=99)
        _, args = mgr.create_model(
            "diffusion",
            {"N_grid_channels": 2},
        )
        assert args["checkpoint_level"] == 99

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_input_channels_include_static(self, MockEDM):
        """img_in_channels should include static channels."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset(n_input=4, n_static=2)
        mgr = _make_manager_corrdiff(dataset=ds, n_month_hour_channels=0)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 3}
        )
        # img_in_channels = n_input(4) + n_static(2) + n_month_hour(0) + N_grid_channels(3)
        assert args["img_in_channels"] == 4 + 2 + 3

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_input_channels_include_month_hour(self, MockEDM):
        """img_in_channels should include month/hour embedding channels."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset(n_input=4, n_static=2)
        mgr = _make_manager_corrdiff(dataset=ds, n_month_hour_channels=6)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 0}
        )
        # img_in_channels = 4 + 2 + 6 + 0
        assert args["img_in_channels"] == 12

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_hr_mean_conditioning_adds_output_channels(self, MockEDM):
        """hr_mean_conditioning should add n_output channels to img_in_channels."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset(n_input=4, n_output=3, n_static=2)
        mgr = _make_manager_corrdiff(dataset=ds, hr_mean_conditioning=True)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 0}
        )
        # img_in_channels = 4 + 2 + 0 (month/hour) + 3 (hr_mean) + 0 (N_grid)
        assert args["img_in_channels"] == 4 + 2 + 3

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_patching_adds_input_and_static_channels(self, MockEDM):
        """use_patching should add an extra set of input + static channels."""
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset(n_input=4, n_static=2, n_output=3)
        mgr = _make_manager_corrdiff(dataset=ds, use_patching=True, n_month_hour_channels=6, hr_mean_conditioning=True)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 5}
        )
        # img_in_channels = (4+2) + (4+2) for patching + 6 (month/hour) + 5 (N_grid) + 3 (hr_mean) = (4+2)*2 + 6 + 5 + 3
        assert args["img_in_channels"] == (4 + 2) * 2 + 6 + 5 + 3

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_amp_mode_set_when_enabled(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, enable_amp=True)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 0}
        )
        assert args["amp_mode"] is True

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_amp_mode_absent_when_disabled(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset()
        mgr = _make_manager_corrdiff(dataset=ds, enable_amp=False)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 0}
        )
        assert "amp_mode" not in args

    @patch("hirad.training.training_manager.EDMPrecondSuperResolution")
    def test_img_in_out_channels_in_model_args(self, MockEDM):
        MockEDM.return_value = MagicMock(spec=nn.Module)
        ds = _make_mock_dataset(n_input=4, n_output=3, n_static=2)
        mgr = _make_manager_corrdiff(dataset=ds, n_month_hour_channels=6)
        _, args = mgr.create_model(
            "diffusion", {"N_grid_channels": 0}
        )
        assert "img_in_channels" in args
        assert "img_out_channels" in args

############################################################################
#          TrainingManagerCorrDiff — load_regression_model                  #
############################################################################


class TestLoadRegressionModel:
    """Tests for TrainingManagerCorrDiff.load_regression_model."""

    def test_missing_dir_raises_file_not_found(self, tmp_path):
        mgr = _make_manager_corrdiff()
        with pytest.raises(FileNotFoundError, match="not found"):
            mgr.load_regression_model(str(tmp_path / "nonexistent"))

    def test_missing_model_args_json_raises(self, tmp_path):
        """Directory exists but model_args.json is missing."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        mgr = _make_manager_corrdiff()
        with pytest.raises(FileNotFoundError, match="model_args.json"):
            mgr.load_regression_model(str(ckpt_dir))

    @patch("hirad.training.training_manager.load_checkpoint")
    @patch("hirad.training.training_manager.UNet")
    def test_loads_and_returns_model(self, MockUNet, mock_load_ckpt, tmp_path):
        """Should load model_args.json and return a UNet in eval mode."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model_args = {
            "img_in_channels": 6,
            "img_out_channels": 3,
            "img_resolution": [64, 64],
        }
        (ckpt_dir / "model_args.json").write_text(json.dumps(model_args))

        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.requires_grad_.return_value = mock_model
        mock_model.to.return_value = mock_model
        MockUNet.return_value = mock_model
        mock_load_ckpt.return_value = 0

        mgr = _make_manager_corrdiff()
        result = mgr.load_regression_model(str(ckpt_dir))

        MockUNet.assert_called_once()
        mock_model.eval.assert_called_once()
        mock_model.requires_grad_.assert_called_once_with(False)
        assert result is mock_model

    @patch("hirad.training.training_manager.load_checkpoint")
    @patch("hirad.training.training_manager.UNet")
    def test_passes_apex_and_profile_and_amp_flags(self, MockUNet, mock_load_ckpt, tmp_path):
        """UNet should receive use_apex_gn, profile_mode, and amp_mode."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model_args.json").write_text(
            json.dumps({"img_in_channels": 6, "img_out_channels": 3, "img_resolution": [64, 64]})
        )
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.requires_grad_.return_value = mock_model
        mock_model.to.return_value = mock_model
        MockUNet.return_value = mock_model
        mock_load_ckpt.return_value = 0

        mgr = _make_manager_corrdiff(use_apex_gn=True, profile_mode=True, enable_amp=True)
        mgr.load_regression_model(str(ckpt_dir))

        call_kwargs = MockUNet.call_args[1]
        assert call_kwargs["use_apex_gn"] is True
        assert call_kwargs["profile_mode"] is True
        assert call_kwargs["amp_mode"] is True

    @patch("hirad.training.training_manager.load_checkpoint")
    @patch("hirad.training.training_manager.UNet")
    def test_apex_gn_sets_channels_last(self, MockUNet, mock_load_ckpt, tmp_path):
        """With use_apex_gn, model.to(memory_format=channels_last) should be called."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model_args.json").write_text(
            json.dumps({"img_in_channels": 6, "img_out_channels": 3, "img_resolution": [64, 64]})
        )
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_model.requires_grad_.return_value = mock_model
        mock_model.to.return_value = mock_model
        MockUNet.return_value = mock_model
        mock_load_ckpt.return_value = 0

        mgr = _make_manager_corrdiff(use_apex_gn=True)
        mgr.load_regression_model(str(ckpt_dir))

        mock_model.to.assert_any_call(memory_format=torch.channels_last)


############################################################################
#            TrainingManagerCorrDiff — run_validation                       #
############################################################################


class TestRunValidation:
    """Tests for TrainingManagerCorrDiff.run_validation."""

    @staticmethod
    def _make_loss_fn(loss_value=1.0, loss_size=B):
        loss_fn = MagicMock()
        loss_fn.return_value = torch.tensor([loss_value] * loss_size)
        loss_fn.y_mean = None
        return loss_fn

    @staticmethod
    def _make_validation_iterator(n_steps, n_out=C_OUT, n_in=C_IN):
        batches = [
            [torch.randn(B, n_out, H * W), torch.randn(B, n_in, H * W)]
            for _ in range(n_steps)
        ]
        return iter(batches)

    def test_returns_float(self):
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(2)
        result = mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=2,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=None,
        )
        assert isinstance(result, float)

    def test_calls_loss_fn_per_step(self):
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        n_steps = 3
        it = self._make_validation_iterator(n_steps)
        mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=n_steps,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=None,
        )
        assert loss_fn.call_count == n_steps

    def test_calls_loss_fn_per_patch_iter(self):
        """Loss should be called validation_steps * len(patch_nums_iter) times."""
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        n_steps = 2
        patch_nums_iter = [2, 2, 1]
        it = self._make_validation_iterator(n_steps)
        patching = MagicMock()
        mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=n_steps,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=patching,
            patch_nums_iter=patch_nums_iter,
            use_patch_grad_acc=None,
        )
        assert loss_fn.call_count == n_steps * len(patch_nums_iter)

    def test_sets_patch_num_on_patching(self):
        """patching.set_patch_num should be called for each patch iteration."""
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(1)
        patching = MagicMock()
        patch_nums_iter = [3, 2]
        mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=patching,
            patch_nums_iter=patch_nums_iter,
            use_patch_grad_acc=None,
        )
        calls = [c.args[0] for c in patching.set_patch_num.call_args_list]
        assert calls == [3, 2]

    @patch("hirad.training.training_manager.mlflow")
    def test_logs_to_mlflow_on_rank0(self, mock_mlflow):
        dist = _make_mock_dist(rank=0, world_size=1)
        mgr = _make_manager_corrdiff(dist=dist, logging_method="mlflow")
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(1)
        mgr.run_validation(
            cur_nimg=200,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=None,
        )
        mock_mlflow.log_metric.assert_called_once()
        call_args = mock_mlflow.log_metric.call_args
        assert call_args[0][0] == "validation_loss"
        assert call_args[0][2] == 200  # cur_nimg

    @patch("hirad.training.training_manager.mlflow")
    def test_no_mlflow_on_non_rank0(self, mock_mlflow):
        dist = _make_mock_dist(rank=1, world_size=1) # keep world size 1 not to trigger any distributed logic, but set rank to non-zero
        mgr = _make_manager_corrdiff(dist=dist, logging_method="mlflow")
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(1)
        mgr.run_validation(
            cur_nimg=200,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=None,
        )
        mock_mlflow.log_metric.assert_not_called()

    @patch("hirad.training.training_manager.mlflow")
    def test_no_mlflow_when_logging_disabled(self, mock_mlflow):
        dist = _make_mock_dist(rank=0, world_size=1)
        mgr = _make_manager_corrdiff(dist=dist, logging_method=None)
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(1)
        mgr.run_validation(
            cur_nimg=200,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=None,
        )
        mock_mlflow.log_metric.assert_not_called()

    def test_resets_y_mean_with_patch_grad_acc(self):
        """When use_patch_grad_acc is True, loss_fn.y_mean should be reset each step."""
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        loss_fn.y_mean = torch.tensor(42.0)
        it = self._make_validation_iterator(1)
        mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=True,
        )
        # y_mean should have been set to None at the beginning of the step
        assert loss_fn.y_mean is None

    def test_average_loss_value_as_expected(self):
        """Test that the average loss value is computed as expected."""
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn()
        it = self._make_validation_iterator(1)
        result = mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=1,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=None,
            patch_nums_iter=[1],
            use_patch_grad_acc=True,
        )
        assert result == 1.0

    def test_average_loss_value_with_patching_as_expected(self):
        """Test that the average loss value is computed as expected when using patching."""
        mgr = _make_manager_corrdiff()
        loss_fn = self._make_loss_fn(loss_size=B*3)  # simulate 3 patches per batch_element
        it = self._make_validation_iterator(3)
        patch_nums_iter = [3, 3]
        result = mgr.run_validation(
            cur_nimg=100,
            validation_dataset_iterator=it,
            model=MagicMock(),
            loss_fn=loss_fn,
            validation_steps=3,
            static_channels=None,
            batch_size_per_gpu=B,
            patching=MagicMock(),
            patch_nums_iter=patch_nums_iter,
            use_patch_grad_acc=True,
        )
        # With 2 total patch iterations with 3 patches per iteration and a loss of 1.0 per iteration, the average should still be 1.0
        assert result == 1.0