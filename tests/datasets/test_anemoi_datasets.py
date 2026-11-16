"""Index mapping, alignment and normalization logic of the anemoi-backed datasets.

``open_dataset`` is replaced by an in-memory fake that mimics the anemoi semantics these
classes rely on: ``select`` reorders data *and* statistics, start/end/frequency/area subset
an analysis store, and trajectory (forecast) stores cannot be subset and keep their native
variable order. Values encode (variable, time, point) so tests can check what was read.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from hirad.datasets import AnemoiDataset, AnemoiForecastDataset
from hirad.datasets import dataset as dataset_module

TARGET_SHAPE = (4, 5)
INPUT_VARS = ["10u", "2t", "tp"]  # native (alphabetical) order in the stores
INPUT_NAMES = ["2t", "tp", "10u"]  # order requested by the config


def _grid(lat_values, lon_values):
    lat, lon = np.meshgrid(lat_values, lon_values, indexing="ij")
    return lat.ravel(), lon.ravel()


TARGET_LAT, TARGET_LON = _grid(np.linspace(46.0, 46.3, TARGET_SHAPE[0]), np.linspace(7.0, 7.4, TARGET_SHAPE[1]))
INPUT_LAT, INPUT_LON = _grid(np.linspace(45.0, 47.5, 6), np.linspace(6.0, 8.5, 6))


def _bound(value, is_end):
    stamp = pd.Timestamp(value)
    if is_end and isinstance(value, str) and len(value) == 10:
        return stamp + pd.Timedelta("1D") - pd.Timedelta("1s")
    return stamp


def _statistics(data, variable_axis):
    axes = tuple(i for i in range(data.ndim) if i != variable_axis)
    return {"mean": data.mean(axis=axes), "stdev": data.std(axis=axes) + 1.0,
            "minimum": data.min(axis=axes), "maximum": data.max(axis=axes)}


class FakeAnalysisStore:
    def __init__(self, dates, variables, data, latitudes, longitudes, field_shape, statistics=None):
        self.dates = np.asarray(pd.DatetimeIndex(dates).values)
        self.variables = list(variables)
        self.data = data  # (date, variable, ensemble, point)
        self.latitudes, self.longitudes, self.field_shape = latitudes, longitudes, field_shape
        self.statistics = statistics if statistics is not None else _statistics(data, 1)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        return self.data[key]

    def metadata(self):
        return {"start_date": str(pd.Timestamp(self.dates[0])), "end_date": str(pd.Timestamp(self.dates[-1]))}

    def subset(self, select=None, start=None, end=None, area=None, frequency=None):
        dates = pd.DatetimeIndex(self.dates)
        keep = np.ones(len(dates), dtype=bool)
        if start is not None:
            keep &= dates >= _bound(start, is_end=False)
        if end is not None:
            keep &= dates <= _bound(end, is_end=True)
        if frequency is not None:
            hours = int(pd.Timedelta(frequency) / pd.Timedelta("1h"))
            keep &= (dates.hour % hours) == 0
        variables = select if select is not None else self.variables
        var_idx = [self.variables.index(v) for v in variables]
        points = np.ones(len(self.latitudes), dtype=bool)
        if area is not None:
            north, west, south, east = area
            points = (self.latitudes <= north) & (self.latitudes >= south) & (self.longitudes >= west) & (self.longitudes <= east)
        stats = {k: v[var_idx] for k, v in self.statistics.items()}
        return FakeAnalysisStore(
            dates[keep], variables, self.data[keep][:, var_idx][..., points],
            self.latitudes[points], self.longitudes[points], self.field_shape, stats,
        )


class FakeTrajectoryStore:
    """5D (base_date, variable, ensemble, step, point); no ``dates`` and no subsetting."""

    def __init__(self, base_dates, steps_hours, variables, data, latitudes, longitudes):
        self.base_dates = np.asarray(pd.DatetimeIndex(base_dates).values)
        self.steps = np.array([np.timedelta64(h, "h") for h in steps_hours])
        self.variables = list(variables)
        self.data = data
        self.latitudes, self.longitudes, self.field_shape = latitudes, longitudes, (len(latitudes),)
        self.statistics = _statistics(data, 1)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        return self.data[key]


def _analysis(dates, variables, latitudes, longitudes, field_shape, offset=0.0):
    dates = pd.DatetimeIndex(dates)
    v = np.arange(len(variables))[None, :, None, None]
    t = np.arange(len(dates))[:, None, None, None]
    p = np.arange(len(latitudes))[None, None, None, :]
    data = (offset + 100.0 * (v + 1) + t + 0.001 * p).astype(np.float32)
    return FakeAnalysisStore(dates, variables, data, latitudes, longitudes, field_shape)


def _fake_open(stores):
    def open_dataset(path, select=None, start=None, end=None, area=None, frequency=None, trim_edge=None):
        store = stores[path]
        if isinstance(store, FakeTrajectoryStore):
            assert select is None and start is None, "trajectory stores cannot be subset"
            return store
        return store.subset(select=select, start=start, end=end, area=area, frequency=frequency)
    return open_dataset


@pytest.fixture
def make_analysis_dataset():
    def make(input_dates, target_dates, **kwargs):
        stores = {
            "input": _analysis(input_dates, INPUT_VARS, INPUT_LAT, INPUT_LON, (len(INPUT_LAT),)),
            "target": _analysis(target_dates, ["2t", "tp", "hsurf"], TARGET_LAT, TARGET_LON, TARGET_SHAPE, offset=1000.0),
        }
        kwargs = {"input_channel_names": INPUT_NAMES, "output_channel_names": ["2t", "tp"],
                  "n_month_hour_channels": 4, **kwargs}
        with patch("hirad.datasets.anemoi_dataset.open_dataset", _fake_open(stores)):
            return AnemoiDataset(type="anemoi_era5_cosmo", input_anemoi_dataset_path="input",
                                 target_anemoi_dataset_path="target", **kwargs), stores
    return make


HOURS_DAY1 = pd.date_range("2020-01-01T00:00", "2020-01-01T23:00", freq="1h")
HOURS_DAY1_2 = pd.date_range("2020-01-01T00:00", "2020-01-02T23:00", freq="1h")


class TestAnalysisAlignment:
    def test_samples_pair_input_and_target_at_same_valid_time(self, make_analysis_dataset):
        ds, stores = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T05:00")
        assert len(ds) == 6
        assert ds.time() == [f"20200101-0{h}00" for h in range(6)]
        target, inputs, date = ds[2]
        assert date == "20200101-0200"
        np.testing.assert_array_equal(target.numpy(), stores["target"].subset(select=["2t", "tp"]).data[2].squeeze())
        assert inputs.shape == (len(INPUT_NAMES), 9)  # area-cropped input points

    def test_missing_target_raises_outside_generation(self, make_analysis_dataset):
        with pytest.raises(ValueError, match="no target at the same valid time"):
            make_analysis_dataset(HOURS_DAY1, HOURS_DAY1.delete(3), start_date="2020-01-01", end_date="2020-01-01T05:00")

    def test_missing_target_is_zeros_for_generation(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1.delete(3), start_date="2020-01-01",
                                      end_date="2020-01-01T05:00", target_missing_as_zeros=True)
        assert len(ds) == 6
        target, _, date = ds[3]
        assert date == "20200101-0300" and not target.any()
        assert ds.previous_index(3) == 2
        assert ds.previous_index(4) is None  # predecessor has no target

    def test_input_frequency_subsamples_input_only(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T23:00",
                                      input_frequency="6h")
        assert ds.time() == ["20200101-0000", "20200101-0600", "20200101-1200", "20200101-1800"]

    def test_exclude_window_drops_samples_and_breaks_history(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1_2, HOURS_DAY1_2, start_date="2020-01-01", end_date="2020-01-02",
                                      exclude_start_date="2020-01-01T06:00", exclude_end_date="2020-01-01T11:00")
        times = ds.time()
        assert len(ds) == 42 and len(times) == 42
        assert "20200101-0600" not in times and "20200101-1100" not in times
        first_after_gap = times.index("20200101-1200")
        assert ds.previous_index(first_after_gap) is None
        assert ds.previous_index(first_after_gap + 1) == first_after_gap
        assert ds.previous_index(0) is None
        assert ds[first_after_gap][2] == "20200101-1200"

    def test_exclude_whole_day_with_date_only_end(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1_2, HOURS_DAY1_2, start_date="2020-01-01", end_date="2020-01-02",
                                      exclude_start_date="2020-01-01", exclude_end_date="2020-01-01")
        assert ds.time()[0] == "20200102-0000" and len(ds) == 24

    def test_previous_target_is_served_when_requested(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T05:00",
                                      provide_prev_target=True)
        target_0 = ds[0][0]
        _, _, _, prev_target, has_prev = ds[1]
        assert has_prev == 1 and torch.equal(prev_target, target_0)
        _, _, _, prev_target, has_prev = ds[0]
        assert has_prev == 0 and not prev_target.any()


class TestAnalysisChannelsAndNormalization:
    def test_select_orders_channels_and_statistics_consistently(self, make_analysis_dataset):
        ds, stores = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T05:00")
        full = stores["input"].statistics
        order = [INPUT_VARS.index(name) for name in INPUT_NAMES]
        np.testing.assert_allclose(ds.input_mean, full["mean"][order])
        np.testing.assert_allclose(ds.input_std, full["stdev"][order])
        _, inputs, _ = ds[0]
        # Fake values encode the native variable index in the hundreds.
        assert [int(inputs[c, 0]) // 100 for c in range(3)] == [INPUT_VARS.index(n) + 1 for n in INPUT_NAMES]
        normalized = ds.normalize_input(inputs.numpy()[None].copy())
        np.testing.assert_allclose(normalized[0, 0], (inputs[0].numpy() - ds.input_mean[0]) / ds.input_std[0], rtol=1e-6)

    def test_box_cox_output_round_trip(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(
            HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T05:00",
            transform_channels=["tp-box_cox_025"],
            transform_input_means={"tp-box_cox_025": -3.8}, transform_input_stdevs={"tp-box_cox_025": 0.2},
            transform_output_means={"tp-box_cox_025": -3.9}, transform_output_stdevs={"tp-box_cox_025": 0.19},
        )
        assert ds.output_mean[1] == pytest.approx(-3.9) and ds.output_std[1] == pytest.approx(0.19)
        x = np.abs(np.random.default_rng(0).standard_normal((1, 2, 20))).astype(np.float64)
        restored = ds.denormalize_output(ds.normalize_output(x.copy()))
        np.testing.assert_allclose(restored, x, rtol=1e-6, atol=1e-9)

    def test_time_grids_encode_hour_and_month(self, make_analysis_dataset):
        ds, _ = make_analysis_dataset(HOURS_DAY1, HOURS_DAY1, start_date="2020-01-01", end_date="2020-01-01T05:00")
        grid = ds.make_time_grids(["20200101-0600", "20200701-0000"], torch.device("cpu"))
        assert grid.shape == (2, 8)
        # hour 6 -> quarter period: sin(2*pi*0.25) = 1 for the first hour frequency
        assert grid[0, 0].item() == pytest.approx(1.0, abs=1e-6)
        # month 7 -> half period: cos(pi) = -1 for the first month frequency
        assert grid[1, 5].item() == pytest.approx(-1.0, abs=1e-6)


BASES = pd.DatetimeIndex(["2020-01-01T00:00", "2020-01-01T12:00", "2020-01-02T00:00"])
STEPS = list(range(1, 7))


@pytest.fixture
def make_forecast_dataset():
    def make(target_dates, **kwargs):
        b = np.arange(len(BASES))[:, None, None, None, None]
        v = np.arange(len(INPUT_VARS))[None, :, None, None, None]
        s = np.array(STEPS)[None, None, None, :, None]
        p = np.arange(len(INPUT_LAT))[None, None, None, None, :]
        data = (100.0 * (v + 1) + 10.0 * b + 0.1 * s + 0.001 * p).astype(np.float32)
        stores = {
            "input": FakeTrajectoryStore(BASES, STEPS, INPUT_VARS, data, INPUT_LAT, INPUT_LON),
            "target": _analysis(target_dates, ["2t", "tp"], TARGET_LAT, TARGET_LON, TARGET_SHAPE, offset=1000.0),
        }
        kwargs = {"input_channel_names": INPUT_NAMES, "output_channel_names": ["2t", "tp"], **kwargs}
        opener = _fake_open(stores)
        with patch("hirad.datasets.anemoi_dataset.open_dataset", opener), \
                patch("hirad.datasets.anemoi_forecast_dataset.open_dataset", opener):
            return AnemoiForecastDataset(type="anemoi_ifs_cosmo", input_anemoi_dataset_path="input",
                                         target_anemoi_dataset_path="target", **kwargs), stores
    return make


FORECAST_TARGET_HOURS = pd.date_range("2020-01-01T00:00", "2020-01-02T06:00", freq="1h")


class TestForecastPairing:
    def test_pairs_respect_init_window_and_max_lead(self, make_forecast_dataset):
        ds, _ = make_forecast_dataset(FORECAST_TARGET_HOURS, start_date="2020-01-01T00:00",
                                      end_date="2020-01-01T12:00", max_lead_hours=4, provide_lead_time=True)
        assert len(ds) == 8
        assert ds.base_time() == ["20200101-0000"] * 4 + ["20200101-1200"] * 4
        assert ds.time()[:4] == ["20200101-0100", "20200101-0200", "20200101-0300", "20200101-0400"]
        target, inputs, date, lead = ds[5]
        assert date == "20200101-1400" and lead == 2

    def test_channels_and_statistics_reordered_from_native_order(self, make_forecast_dataset):
        ds, stores = make_forecast_dataset(FORECAST_TARGET_HOURS, start_date="2020-01-01T00:00",
                                           end_date="2020-01-01T12:00")
        order = [INPUT_VARS.index(name) for name in INPUT_NAMES]
        np.testing.assert_allclose(ds.input_mean, stores["input"].statistics["mean"][order])
        np.testing.assert_allclose(ds.input_std, stores["input"].statistics["stdev"][order])
        _, inputs, _ = ds[0]
        assert [int(inputs[c, 0]) // 100 for c in range(3)] == [INPUT_VARS.index(n) + 1 for n in INPUT_NAMES]

    def test_previous_index_stays_within_one_run(self, make_forecast_dataset):
        ds, _ = make_forecast_dataset(FORECAST_TARGET_HOURS, start_date="2020-01-01T00:00",
                                      end_date="2020-01-01T12:00", max_lead_hours=3)
        assert [ds.previous_index(i) for i in range(len(ds))] == [None, 0, 1, None, 3, 4]

    def test_leads_past_target_are_dropped_for_training(self, make_forecast_dataset):
        short_target = pd.date_range("2020-01-01T00:00", "2020-01-01T14:00", freq="1h")
        ds, _ = make_forecast_dataset(short_target, start_date="2020-01-01T00:00", end_date="2020-01-01T12:00")
        assert ds.base_time().count("20200101-1200") == 2  # only leads 1-2 have a target

    def test_missing_targets_are_zeros_for_generation(self, make_forecast_dataset):
        short_target = pd.date_range("2020-01-01T00:00", "2020-01-01T14:00", freq="1h")
        ds, _ = make_forecast_dataset(short_target, start_date="2020-01-01T00:00", end_date="2020-01-01T12:00",
                                      target_missing_as_zeros=True)
        assert len(ds) == 12
        target, _, date = ds[11]
        assert date == "20200101-1800" and not target.any()


class TestInferenceTimeSelection:
    def test_every_matching_pair_is_selected_and_unknown_times_rejected(self, make_forecast_dataset):
        ds, _ = make_forecast_dataset(FORECAST_TARGET_HOURS, start_date="2020-01-01T00:00", end_date="2020-01-02T00:00")
        with patch.object(dataset_module, "init_dataset_from_config", return_value=(ds, None)):
            _, sampler = dataset_module.get_dataset_and_sampler_inference({}, times=["20200101-1300"])
            assert [ds.base_time()[i] for i in sampler] == ["20200101-1200"]
            _, sampler = dataset_module.get_dataset_and_sampler_inference({}, times=["20200102-0100"])
            assert sorted(ds.base_time()[i] for i in sampler) == ["20200102-0000"]
            with pytest.raises(ValueError, match="not found"):
                dataset_module.get_dataset_and_sampler_inference({}, times=["20300101-0000"])
