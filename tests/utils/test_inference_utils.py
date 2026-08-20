import pytest
import os
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from hirad.datasets.base import ChannelMetadata
from hirad.utils.inference_utils import (
    calculate_bounds,
    regression_step,
    diffusion_step,
    save_results_as_torch,
    forecast_run_key_and_step,
    accumulate_tp_channel,
    pad_image,
    get_grib_template,
    save_image_as_grib,
    save_results_as_grib,
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


############################################################################
#                       forecast_run_key_and_step                          #
############################################################################


class TestForecastRunKeyAndStep:
    """Tests for forecast_run_key_and_step."""

    def test_with_base_time_computes_hour_offset(self):
        run_key, step_num = forecast_run_key_and_step("20230115-0300", base_time="20230115-0000")
        assert run_key == "20230115-0000"
        assert step_num == 3

    def test_with_base_time_across_days(self):
        run_key, step_num = forecast_run_key_and_step("20230115-0600", base_time="20230114-1800")
        assert run_key == "20230114-1800"
        assert step_num == 12

    def test_with_base_time_equal_to_time_step_is_step_zero(self):
        run_key, step_num = forecast_run_key_and_step("20230115-0000", base_time="20230115-0000")
        assert run_key == "20230115-0000"
        assert step_num == 0

    def test_without_base_time_derives_run_key_and_step_from_time_step(self):
        run_key, step_num = forecast_run_key_and_step("20230115-0300")
        assert run_key == "20230115"
        assert step_num == 3

    def test_without_base_time_two_digit_hour(self):
        run_key, step_num = forecast_run_key_and_step("20230115-2300")
        assert run_key == "20230115"
        assert step_num == 23


############################################################################
#                          accumulate_tp_channel                           #
############################################################################


class TestAccumulateTpChannel:
    """Tests for accumulate_tp_channel."""

    def test_first_step_running_total_equals_input(self):
        pred = np.zeros((3, 2, 2))
        pred[1] = 5.0  # tp is channel index 1
        cumulative: dict = {}

        accumulate_tp_channel(pred, tp_idx=1, run_key="run_a", step_num=0, cumulative_precip=cumulative)

        np.testing.assert_array_equal(pred[1], np.full((2, 2), 5.0))
        last_step, total = cumulative["run_a"]
        assert last_step == 0
        np.testing.assert_array_equal(total, np.full((2, 2), 5.0))

    def test_accumulates_across_sequential_steps(self):
        cumulative: dict = {}
        step0 = np.zeros((3, 2, 2)); step0[1] = 1.0
        step1 = np.zeros((3, 2, 2)); step1[1] = 2.0

        accumulate_tp_channel(step0, 1, "run_a", 0, cumulative)
        accumulate_tp_channel(step1, 1, "run_a", 1, cumulative)

        np.testing.assert_array_equal(step1[1], np.full((2, 2), 3.0))
        last_step, total = cumulative["run_a"]
        assert last_step == 1
        np.testing.assert_array_equal(total, np.full((2, 2), 3.0))

    def test_out_of_order_step_raises(self):
        cumulative: dict = {}
        step0 = np.zeros((3, 2, 2)); step0[1] = 1.0
        step2 = np.zeros((3, 2, 2)); step2[1] = 1.0

        accumulate_tp_channel(step0, 1, "run_a", 0, cumulative)
        with pytest.raises(RuntimeError, match="increasing order"):
            accumulate_tp_channel(step2, 1, "run_a", 2, cumulative)

    def test_repeated_step_raises(self):
        cumulative: dict = {}
        step0 = np.zeros((3, 2, 2)); step0[1] = 1.0

        accumulate_tp_channel(step0, 1, "run_a", 0, cumulative)
        with pytest.raises(RuntimeError, match="increasing order"):
            accumulate_tp_channel(step0.copy(), 1, "run_a", 0, cumulative)

    def test_independent_run_keys_dont_interfere(self):
        cumulative: dict = {}
        a0 = np.zeros((3, 2, 2)); a0[1] = 1.0
        b0 = np.zeros((3, 2, 2)); b0[1] = 100.0

        accumulate_tp_channel(a0, 1, "run_a", 0, cumulative)
        accumulate_tp_channel(b0, 1, "run_b", 0, cumulative)
        b1 = np.zeros((3, 2, 2)); b1[1] = 50.0
        accumulate_tp_channel(b1, 1, "run_b", 1, cumulative)

        assert cumulative["run_a"][0] == 0
        np.testing.assert_array_equal(cumulative["run_a"][1], np.full((2, 2), 1.0))
        assert cumulative["run_b"][0] == 1
        np.testing.assert_array_equal(cumulative["run_b"][1], np.full((2, 2), 150.0))

    def test_other_channels_untouched(self):
        cumulative: dict = {}
        pred = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
        other_channels_before = pred[[0, 2]].copy()

        accumulate_tp_channel(pred, 1, "run_a", 0, cumulative)

        np.testing.assert_array_equal(pred[[0, 2]], other_channels_before)

    def test_ensemble_batch_axis_is_second_to_last_minus_two(self):
        # shape (ensemble, channels, H, W): the channel axis is 1, not 0.
        pred = np.zeros((2, 3, 2, 2))
        pred[0, 1] = 1.0
        pred[1, 1] = 2.0
        cumulative: dict = {}

        accumulate_tp_channel(pred, tp_idx=1, run_key="run_a", step_num=0, cumulative_precip=cumulative)

        np.testing.assert_array_equal(pred[0, 1], np.full((2, 2), 1.0))
        np.testing.assert_array_equal(pred[1, 1], np.full((2, 2), 2.0))


############################################################################
#                                pad_image                                 #
############################################################################


class TestPadImage:
    """Tests for pad_image."""

    def test_output_shape(self):
        image = np.ones((4, 5))
        result = pad_image(image, 3, np.nan)
        assert result.shape == (10, 11)

    def test_center_values_preserved(self):
        image = np.arange(12).reshape(3, 4).astype(float)
        result = pad_image(image, 2, -1.0)
        np.testing.assert_array_equal(result[2:-2, 2:-2], image)

    def test_border_filled_with_fill_value(self):
        image = np.ones((3, 3))
        result = pad_image(image, 2, -999.0)
        result[2:-2, 2:-2] = -999.0  # blank out the center; only border should remain
        assert np.all(result == -999.0)

    def test_nan_fill_value(self):
        image = np.zeros((2, 2))
        result = pad_image(image, 1, np.nan)
        assert np.isnan(result[0, 0])
        assert not np.isnan(result[1, 1])


############################################################################
#                            get_grib_template                             #
############################################################################


class _FakeField:
    """Stand-in for an earthkit-data GRIB field; only .metadata() / .message() are used."""

    def __init__(self, metadata, message=b"FAKE-GRIB-MESSAGE"):
        self._metadata = metadata
        self._message = message

    def metadata(self, key):
        return self._metadata[key]

    def message(self):
        return self._message


class _FakeFieldList:
    """Stand-in for an earthkit-data FieldList; only indexing and .metadata(key) as a
    per-field list are used by get_grib_template."""

    def __init__(self, fields):
        self._fields = fields

    def __getitem__(self, idx):
        return self._fields[idx]

    def metadata(self, key):
        return [f.metadata(key) for f in self._fields]


def _make_ekd_from_source(sfc_fields=(), pl_fields=(), extra_files=None):
    """Build a stand-in for ekd.from_source, keyed by the template file's basename."""
    extra_files = extra_files or {}

    def _from_source(kind, path):
        assert kind == "file"
        basename = os.path.basename(path)
        if basename == "ifs-levtype=sfc.grib":
            return _FakeFieldList(list(sfc_fields))
        if basename == "ifs-levtype=pl.grib":
            return _FakeFieldList(list(pl_fields))
        if basename in extra_files:
            return extra_files[basename]
        raise FileNotFoundError(basename)

    return _from_source


class TestGetGribTemplate:
    """Tests for get_grib_template."""

    def test_tp_channel_uses_tot_prec_template_and_cumulative_window(self):
        tp_field = _FakeField({"shortName": "tp"})
        from_source = _make_ekd_from_source(
            extra_files={"co2-shortName=TOT_PREC.grib": _FakeFieldList([tp_field])}
        )
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="tp")
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=0, step_num=3, grid="co2")

        assert result is not None
        template_field, keys = result
        assert template_field is tp_field
        # cumulative-from-start window: [0, step_num], not [step_num-1, step_num]
        assert keys == {
            "dataDate": 20230101, "dataTime": 0,
            "step": 3, "startStep": 0, "endStep": 3,
        }

    def test_surface_channel_uses_matching_typeoflevel_template(self):
        sfc_field = _FakeField({
            "shortName": "2t", "typeOfLevel": "heightAboveGround",
            "level": 2, "paramId": 167,
        })
        template_field = _FakeField({})
        from_source = _make_ekd_from_source(
            sfc_fields=[sfc_field],
            extra_files={"co2-typeOfLevel=heightAboveGround.grib": _FakeFieldList([template_field])},
        )
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="2t")
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=1200, step_num=6, grid="co2")

        assert result is not None
        field, keys = result
        assert field is template_field
        assert keys == {
            "paramId": 167, "level": 2,
            "dataDate": 20230101, "dataTime": 1200,
            "step": 6, "startStep": 6, "endStep": 6,
        }

    def test_surface_channel_missing_grid_template_file_returns_none(self):
        sfc_field = _FakeField({
            "shortName": "2t", "typeOfLevel": "heightAboveGround",
            "level": 2, "paramId": 167,
        })
        # No extra_files entry for the typeOfLevel file -> FileNotFoundError inside
        # get_grib_template, which should be caught and turned into a skip.
        from_source = _make_ekd_from_source(sfc_fields=[sfc_field])
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="2t")
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=0, step_num=0, grid="co2")

        assert result is None

    def test_pressure_level_channel_uses_isobaric_template(self):
        pl_field = _FakeField({"shortName": "t", "paramId": 130})
        template_field = _FakeField({})
        from_source = _make_ekd_from_source(
            pl_fields=[pl_field],
            extra_files={"co1e-typeOfLevel=isobaricInhPa.grib": _FakeFieldList([template_field])},
        )
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="t", level="850")
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=0, step_num=12, grid="co1e")

        assert result is not None
        field, keys = result
        assert field is template_field
        assert keys == {
            "paramId": 130, "level": 850,
            "dataDate": 20230101, "dataTime": 0,
            "step": 12, "startStep": 12, "endStep": 12,
        }

    def test_pressure_level_channel_without_level_is_not_matched(self):
        # A channel whose name is in the pl index but has no level set should not be
        # (mis)treated as a pressure-level field -- it should fall through to "not found".
        pl_field = _FakeField({"shortName": "t", "paramId": 130})
        from_source = _make_ekd_from_source(pl_fields=[pl_field])
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="t")  # level="" by default
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=0, step_num=0, grid="co2")

        assert result is None

    def test_unknown_channel_returns_none(self):
        from_source = _make_ekd_from_source()
        with patch("hirad.utils.inference_utils.ekd.from_source", side_effect=from_source):
            channel = ChannelMetadata(name="not_a_real_channel")
            result = get_grib_template("tpl", channel, ref_date=20230101, ref_time=0, step_num=0, grid="co2")

        assert result is None


############################################################################
#                            save_image_as_grib                            #
############################################################################


class TestSaveImageAsGrib:
    """Tests for save_image_as_grib."""

    def test_invalid_grid_raises_and_writes_nothing(self, tmp_path):
        out_file = tmp_path / "out.grib"
        with pytest.raises(ValueError, match="co1e and co2"):
            save_image_as_grib(
                str(out_file), 20230101, 0, 0, "tpl",
                [ChannelMetadata(name="2t")], [], np.zeros((1, 2, 2)),
                np.zeros((1, 2, 2)), grid="bogus",
            )
        assert not out_file.exists()

    @patch("hirad.utils.inference_utils.eccodes")
    @patch("hirad.utils.inference_utils.get_grib_template")
    def test_writes_one_message_per_matched_channel(self, mock_get_template, mock_eccodes, tmp_path):
        fake_field = MagicMock()
        fake_field.message.return_value = b"FAKE"
        mock_get_template.side_effect = [
            (fake_field, {"paramId": 167, "level": 2}),
            (fake_field, {"paramId": 134, "level": 0}),
        ]
        mock_eccodes.codes_new_from_message.return_value = "grib-id"

        channels = [ChannelMetadata(name="2t"), ChannelMetadata(name="sp")]
        image = np.random.rand(2, 4, 4)
        out_file = tmp_path / "out.grib"

        save_image_as_grib(
            str(out_file), 20230101, 0, 3, "tpl", channels, [],
            image, np.zeros((1, 4, 4)), grid="co2",
        )

        assert mock_eccodes.codes_new_from_message.call_count == 2
        assert mock_eccodes.codes_write.call_count == 2
        assert mock_eccodes.codes_release.call_count == 2

    @patch("hirad.utils.inference_utils.eccodes")
    @patch("hirad.utils.inference_utils.get_grib_template")
    def test_skips_channel_with_no_template(self, mock_get_template, mock_eccodes, tmp_path):
        found_field = MagicMock()
        found_field.message.return_value = b"FAKE"
        mock_get_template.side_effect = [None, (found_field, {"level": 0})]
        mock_eccodes.codes_new_from_message.return_value = "grib-id"

        channels = [ChannelMetadata(name="missing"), ChannelMetadata(name="sp")]
        image = np.random.rand(2, 4, 4)
        out_file = tmp_path / "out.grib"

        save_image_as_grib(
            str(out_file), 20230101, 0, 0, "tpl", channels, [],
            image, np.zeros((1, 4, 4)), grid="co2",
        )

        assert mock_eccodes.codes_new_from_message.call_count == 1
        assert mock_eccodes.codes_write.call_count == 1

    @patch("hirad.utils.inference_utils.eccodes")
    @patch("hirad.utils.inference_utils.get_grib_template")
    def test_grib_id_released_even_if_set_values_fails(self, mock_get_template, mock_eccodes, tmp_path):
        fake_field = MagicMock()
        fake_field.message.return_value = b"FAKE"
        mock_get_template.return_value = (fake_field, {"level": 0})
        mock_eccodes.codes_new_from_message.return_value = "grib-id"
        mock_eccodes.codes_set_values.side_effect = RuntimeError("boom")

        channels = [ChannelMetadata(name="2t")]
        image = np.random.rand(1, 4, 4)
        out_file = tmp_path / "out.grib"

        with pytest.raises(RuntimeError, match="boom"):
            save_image_as_grib(
                str(out_file), 20230101, 0, 0, "tpl", channels, [],
                image, np.zeros((1, 4, 4)), grid="co2",
            )

        mock_eccodes.codes_release.assert_called_once_with("grib-id")

    @patch("hirad.utils.inference_utils.eccodes")
    @patch("hirad.utils.inference_utils.get_grib_template")
    def test_pads_image_and_replaces_nan_with_missing_value(self, mock_get_template, mock_eccodes, tmp_path):
        fake_field = MagicMock()
        fake_field.message.return_value = b"FAKE"
        mock_get_template.return_value = (fake_field, {"level": 0})
        mock_eccodes.codes_new_from_message.return_value = "grib-id"

        channels = [ChannelMetadata(name="2t")]
        image = np.full((1, 2, 2), 5.0)
        out_file = tmp_path / "out.grib"

        save_image_as_grib(
            str(out_file), 20230101, 0, 0, "tpl", channels, [],
            image, np.zeros((1, 2, 2)), grid="co2",  # co2 -> padding_margin=19
        )

        flat = mock_eccodes.codes_set_values.call_args[0][1]
        expected_len = (2 + 2 * 19) * (2 + 2 * 19)
        assert flat.shape == (expected_len,)
        # padding is NaN -> replaced with the missing-value sentinel; only the
        # original 2x2 pixels keep the real value.
        assert (flat == 9999.0).sum() == expected_len - 4
        assert (flat == 5.0).sum() == 4

        set_calls = {c.args[1]: c.args[2] for c in mock_eccodes.codes_set.call_args_list if len(c.args) >= 3}
        assert set_calls["bitmapPresent"] == 1
        assert set_calls["missingValue"] == 9999.0

    @patch("hirad.utils.inference_utils.eccodes")
    @patch("hirad.utils.inference_utils.get_grib_template")
    def test_grib_keys_applied_after_clone(self, mock_get_template, mock_eccodes, tmp_path):
        fake_field = MagicMock()
        fake_field.message.return_value = b"FAKE"
        keys = {"paramId": 167, "level": 2, "step": 3}
        mock_get_template.return_value = (fake_field, keys)
        mock_eccodes.codes_new_from_message.return_value = "grib-id"

        channels = [ChannelMetadata(name="2t")]
        image = np.zeros((1, 2, 2))
        out_file = tmp_path / "out.grib"

        save_image_as_grib(
            str(out_file), 20230101, 0, 3, "tpl", channels, [],
            image, np.zeros((1, 2, 2)), grid="co2",
        )

        for key, val in keys.items():
            mock_eccodes.codes_set.assert_any_call("grib-id", key, val)


############################################################################
#                           save_results_as_grib                           #
############################################################################


class TestSaveResultsAsGrib:
    """Tests for save_results_as_grib (orchestration around save_image_as_grib)."""

    def _make_dataset(self, output_channels=None, static_channels=None, static_data=None):
        dataset = MagicMock()
        dataset.output_channels.return_value = output_channels or [ChannelMetadata(name="2t")]
        dataset.input_channels.return_value = []
        dataset.static_channels.return_value = static_channels or []
        dataset.get_static_data.return_value = (
            static_data if static_data is not None else np.zeros((1, 4, 4))
        )
        return dataset

    @patch("hirad.utils.inference_utils.save_image_as_grib")
    def test_forecast_output_filename_uses_base_time_and_step(self, mock_save_image, tmp_path):
        dataset = self._make_dataset()
        prediction = np.zeros((1, 4, 4))

        save_results_as_grib(
            str(tmp_path), "20230115-0300", np.zeros((1, 4, 4)), prediction, None, None,
            dataset, "tpl", base_time="20230115-0000",
        )

        args = mock_save_image.call_args[0]
        assert args[0] == str(tmp_path / "202301150000_3.grib")
        assert (args[1], args[2], args[3]) == (20230115, 0, 3)

    @patch("hirad.utils.inference_utils.save_image_as_grib")
    def test_reanalysis_output_filename_fakes_base_time_from_time_step(self, mock_save_image, tmp_path):
        dataset = self._make_dataset()
        prediction = np.zeros((1, 4, 4))

        save_results_as_grib(
            str(tmp_path), "20230115-0600", np.zeros((1, 4, 4)), prediction, None, None,
            dataset, "tpl", base_time=None,
        )

        args = mock_save_image.call_args[0]
        assert args[0] == str(tmp_path / "202301150000_6.grib")
        assert (args[1], args[2], args[3]) == (20230115, 0, 6)

    @patch("hirad.utils.inference_utils.save_image_as_grib")
    def test_only_first_ensemble_member_written(self, mock_save_image, tmp_path):
        dataset = self._make_dataset()
        prediction = np.stack([np.full((1, 4, 4), 1.0), np.full((1, 4, 4), 2.0)])  # (2,1,4,4)

        save_results_as_grib(
            str(tmp_path), "20230115-0300", np.zeros((1, 4, 4)), prediction, None, None,
            dataset, "tpl", base_time="20230115-0000",
        )

        written_image = mock_save_image.call_args[0][7]
        np.testing.assert_array_equal(written_image, np.full((1, 4, 4), 1.0))

    @pytest.mark.parametrize("height,grid", [(352, "co2"), (400, "co1e")])
    @patch("hirad.utils.inference_utils.save_image_as_grib")
    def test_grid_selected_from_target_shape(self, mock_save_image, height, grid, tmp_path):
        dataset = self._make_dataset()
        target = np.zeros((1, height, 4))
        prediction = np.zeros((1, height, 4))

        save_results_as_grib(
            str(tmp_path), "20230115-0300", target, prediction, None, None,
            dataset, "tpl", base_time="20230115-0000",
        )

        assert mock_save_image.call_args.kwargs["grid"] == grid

    @patch("hirad.utils.inference_utils.save_image_as_grib")
    def test_channels_and_static_data_passed_through(self, mock_save_image, tmp_path):
        channels = [ChannelMetadata(name="2t"), ChannelMetadata(name="10u")]
        static = [ChannelMetadata(name="orog")]
        static_data = np.arange(16).reshape(1, 4, 4).astype(float)
        dataset = self._make_dataset(output_channels=channels, static_channels=static, static_data=static_data)
        prediction = np.zeros((2, 4, 4))

        save_results_as_grib(
            str(tmp_path), "20230115-0300", np.zeros((2, 4, 4)), prediction, None, None,
            dataset, "tpl", base_time="20230115-0000",
        )

        args = mock_save_image.call_args[0]
        assert args[5] == channels
        assert args[6] == static
        np.testing.assert_array_equal(args[8], static_data)