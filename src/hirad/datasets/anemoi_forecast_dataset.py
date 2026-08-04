from .anemoi_dataset import AnemoiDataset

from anemoi.datasets import open_dataset
import datetime
import numpy as np
from pandas import to_datetime
import torch
from typing import List


class AnemoiForecastDataset(AnemoiDataset):
    """Forecast variant of AnemoiDataset, built on the 5-dimension anemoi-datasets
    layout (anemoi-datasets >= 0.5.38) where the input dataset carries an extra
    step/lead-time axis: shape (base_dates, variables, ensemble, step, values),
    indexed as dataset[ref_idx, :, :, step_idx, :] for a single (reference_time,
    step) pair. Reference times come from `.base_dates`/`.base_date(idx)`
    instead of `.dates`, and lead times from `.steps`.

    One dataset item = one (reference_time, step) pair, i.e. each forecast step
    is downscaled independently against the target dataset's matching valid time
    (reference_time + step).

    TODO: assumes every (base_date, step) valid time exists in the target
    dataset's dates. Add handling (skip/filter, like IFSDataset's
    _target_indices/_aligned_dates) for valid times that fall outside the
    target's coverage or don't land on one of its timestamps.

    Inherited unchanged from AnemoiDataset (same target/static grid and channel
    definitions; unaffected by the input's extra step axis): longitude(),
    latitude(), input_channels(), output_channels(), static_channels(),
    image_shape(), input_shape(), get_static_data(), normalization_stats(),
    stats_to_torch()/stats_to_numpy(), normalize/denormalize_input()/output(),
    box_cox_transform()/box_cox_inverse_transform(), make_time_grids() (operates
    on the 'YYYYMMDD-HHMM' valid-time strings returned by time(), whose format
    is unchanged here).

    Overridden here (touch the reference-time/step structure directly):
    _align_input_output(), __getitem__, __len__, time().

    `type` follows "anemoi_ifsn320_<cosmo|real>" - the input is IFS forecast data
    on the N320 grid, not era5, hence the distinct VALID_INPUT_DATASETS name.
    """

    VALID_INPUT_DATASETS = {'ifsn320'}

    def __init__(self,
                type: str,
                input_anemoi_dataset_path: str,
                target_anemoi_dataset_path: str,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None,
                input_channel_names: List[str] = [],
                output_channel_names: List[str] = [],
                static_channel_names: List[str] = [],
                transform_channels: List[str] = [],
                transform_input_means: dict = {},
                transform_input_stdevs: dict = {},
                transform_output_means: dict = {},
                transform_output_stdevs: dict = {},
                n_month_hour_channels: int = None,
                trim_edge: int = 0,
                ):
        super().__init__(
            type=type,
            input_anemoi_dataset_path=input_anemoi_dataset_path,
            target_anemoi_dataset_path=target_anemoi_dataset_path,
            start_date=start_date,
            end_date=end_date,
            input_channel_names=input_channel_names,
            output_channel_names=output_channel_names,
            static_channel_names=static_channel_names,
            transform_channels=transform_channels,
            transform_input_means=transform_input_means,
            transform_input_stdevs=transform_input_stdevs,
            transform_output_means=transform_output_means,
            transform_output_stdevs=transform_output_stdevs,
            n_month_hour_channels=n_month_hour_channels,
            trim_edge=trim_edge,
        )

    def _open_input_dataset(self, input_anemoi_dataset_path, input_channel_names, start_date, end_date, area):
        # start/end subsetting is unusable here: anemoi-datasets' date-based
        # subsetting (used by the base class) reads a top-level `dates` array off
        # the zarr store to convert start/end into indices, but this forecast
        # store only has `base_dates`/`steps`, so that lookup raises AttributeError.
        # TODO: restrict by base_dates range once anemoi-datasets supports it for
        # 5D forecast stores (or filter post-hoc in _align_input_output).
        return open_dataset(input_anemoi_dataset_path, select=input_channel_names, area=area)

    def _align_input_output(self):
        """Build one (ref_idx, step_idx, target_idx) entry per (reference_time,
        step) pair, matching each pair's valid time (base_date + step) to its
        index in the target dataset's dates.
        """
        base_dates = to_datetime(self._input_dataset.base_dates)
        steps = self._input_dataset.steps
        target_dates = to_datetime(self._output_dataset.dates)

        self._pairs = []
        for ref_idx, base_date in enumerate(base_dates):
            for step_idx, step in enumerate(steps):
                valid_time = base_date + step
                # TODO: see class docstring - assumes valid_time is always present.
                target_idx = np.nonzero(target_dates == valid_time)[0]
                assert len(target_idx) == 1, \
                    f"Expected exactly one target match for valid_time={valid_time} (base_date={base_date}, step={step}), found {len(target_idx)}."
                self._pairs.append((ref_idx, step_idx, int(target_idx[0])))

    def __getitem__(self, idx):
        """Get input and target data for one (reference_time, step) pair.
        Transform and normalize, but do not interpolate."""

        ref_idx, step_idx, target_idx = self._pairs[idx]
        date_str = to_datetime(self._input_dataset.base_dates[ref_idx] + self._input_dataset.steps[step_idx]).strftime('%Y%m%d-%H%M')

        # Don't reshape, but do squeeze ensemble dimension.
        input_data = self._input_dataset[ref_idx, :, :, step_idx, :].squeeze()

        # Pull target data, squeeze the ensemble dimension
        target_data = self._output_dataset[target_idx].squeeze()

        return torch.from_numpy(target_data.copy()),\
                torch.from_numpy(input_data),\
                date_str

    def __len__(self):
        return len(self._pairs)

    def time(self) -> List:
        """Get valid-time ('YYYYMMDD-HHMM') values, one per (reference_time, step) pair."""
        return [
            to_datetime(self._input_dataset.base_dates[ref_idx] + self._input_dataset.steps[step_idx]).strftime('%Y%m%d-%H%M')
            for ref_idx, step_idx, _ in self._pairs
        ]

ANEMOI_IFSN320_REAL = AnemoiForecastDataset
