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

    Most methods inherited from AnemoiDataset.

    NOTE: This class can currently only be used for inference-only tasks.
    TODO: Add proper handling which does target data pairing (when config
    specifies that this is not inference-only). Will need new config variable.

    `type` follows "anemoi_ifsn320_real" - the input is IFS forecast data
    on the N320 grid, not era5, hence the distinct VALID_INPUT_DATASETS name.
    Could be extended to COSMO grid if needed.
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
        # No select/start/end/area here: anemoi-datasets' Select/date/Cropping
        # subsetting wrappers don't forward the 5D-specific `base_dates`/`steps`
        # attributes (and Cropping/date-subsetting outright fail on a 5D forecast
        # store - see _align_input_output). Opened bare, this assumes the store
        # was already built with exactly input_channel_names (see the assert in
        # AnemoiDataset.__init__) and covering only the needed area/dates.
        return open_dataset(input_anemoi_dataset_path)

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
                # no ground-truth target data; only use for shape/static fields.
                # TODO: Update this to explicitly handle an inference-only case, where target data is missing.
                target_idx = [0]
                #target_idx = np.nonzero(target_dates == valid_time)[0]
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
