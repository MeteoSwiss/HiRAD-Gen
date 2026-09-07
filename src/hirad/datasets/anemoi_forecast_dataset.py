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

    Also usable for training: passing start_date/end_date restricts which
    (base_date, step) pairs are included by valid time (see
    _align_input_output), giving a train/validation split. Unlike AnemoiDataset,
    the target dataset is always opened over its full range here rather than
    restricted to start_date/end_date, since the input can't be date-restricted
    at open time (see _open_input_dataset) - restricting the target instead
    would make in-range input pairs look "missing".

    `type` follows "anemoi_ifsn320_real" or "anemoi_ifso1280_real" - the input is
    IFS forecast data on the N320 grid or the native ~9km O1280 octahedral
    reduced-Gaussian grid (IFS HRES), not era5, hence the distinct
    VALID_INPUT_DATASETS names. Could be extended to COSMO grid if needed.
    """

    VALID_INPUT_DATASETS = {'ifsn320', 'ifso1280'}

    def __init__(self,
                type: str,
                input_anemoi_dataset_path: str,
                target_anemoi_dataset_path: str,
                input_stats_anemoi_dataset_path: str = None,
                target_stats_anemoi_dataset_path: str = None,
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
                target_missing_as_zeros: bool = False,
                input_frequency: str = None,
                lead_time_hours: List[int] = None,
                ):
        # Stashed for _align_input_output to filter pairs by valid time. Not forwarded to
        # the base class: unlike AnemoiDataset, the target here must stay open over its
        # full range (see _open_input_dataset - the input can't be date-restricted at open
        # time, so restricting the target instead would make in-range input pairs look
        # "missing").
        self._split_start_date = to_datetime(start_date) if start_date is not None else None
        self._split_end_date = to_datetime(end_date) if end_date is not None else None
        # Restrict which forecast steps are used (e.g. [1] for 1h-lead-time-only training).
        # None (default) keeps every step, as before.
        self._lead_time_hours = set(lead_time_hours) if lead_time_hours is not None else None
        super().__init__(
            type=type,
            input_anemoi_dataset_path=input_anemoi_dataset_path,
            target_anemoi_dataset_path=target_anemoi_dataset_path,
            input_stats_anemoi_dataset_path=input_stats_anemoi_dataset_path,
            target_stats_anemoi_dataset_path=target_stats_anemoi_dataset_path,
            start_date=None,
            end_date=None,
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
            target_missing_as_zeros=target_missing_as_zeros,
            input_frequency=input_frequency,
        )

    def _open_input_dataset(self, input_anemoi_dataset_path, input_channel_names, start_date, end_date, area, open_dataset_kwargs):
        # No select/start/end/area here: anemoi-datasets' Select/date/Cropping
        # subsetting wrappers don't forward the 5D-specific `base_dates`/`steps`
        # attributes (and Cropping/date-subsetting outright fail on a 5D forecast
        # store - see _align_input_output). Opened bare, this assumes the store
        # covers only the needed area/dates, but NOT that its native variable
        # order matches input_channel_names - forecast stores have been observed
        # to store variables alphabetically rather than in the requested order.
        # Map input_channel_names to native positions and reorder explicitly in
        # __getitem__ (self._channel_indices) instead of trusting the order.
        dataset = open_dataset(input_anemoi_dataset_path)
        native_names = list(dataset.variables)
        missing = [name for name in input_channel_names if name not in native_names]
        if missing:
            raise ValueError(
                f"input_channel_names {missing} not found among the forecast store's "
                f"native variables {native_names}."
            )
        self._channel_indices = [native_names.index(name) for name in input_channel_names]
        return dataset

    def _fallback_input_statistics(self, input_channel_names):
        # The trajectory store can't be opened with select= (see _open_input_dataset), so
        # its native-order statistics must be reordered to match input_channel_names, same
        # as the data itself in __getitem__.
        stats = self._input_dataset.statistics
        return {k: v[self._channel_indices] for k, v in stats.items()}

    def _align_input_output(self):
        """
        Align input and output samples by valid time.
        
        By default, iteration is driven by the input dataset and each input
        (base_time, step) is mapped to the target index at the same
        valid_time (where valid_time = base_time + step). This supports
        different input/target frequencies (e.g. 6h input with an
        hourly target, where only the matching target times are used) as well as
        targets missing some dates. Pairs whose valid time falls outside
        [self._split_start_date, self._split_end_date] (set from the
        start_date/end_date constructor args, when given) are excluded - this is
        how train/validation splits are implemented for this class. Steps whose
        lead time isn't in self._lead_time_hours (set from the lead_time_hours
        constructor arg, when given) are excluded entirely. each member of
        `_pairs` is a tuple (base_date_idx, step_idx, target_idx) where
        target_idx is the index of the matching target or None when the target
        has no matching date.
        """
        base_dates = to_datetime(self._input_dataset.base_dates)
        steps = self._input_dataset.steps
        target_dates = to_datetime(self._output_dataset.dates)

        # Note: assumes output dataset is a reanalysis dataset, not a forecast dataset.
        output_date_to_idx = {
            to_datetime(d): i for i, d in enumerate(self._output_dataset.dates)
        }

        self._pairs = []

        # Shape of a single (pre-squeeze) target sample, used to emit zeros when missing.
        self._target_sample_shape = self._output_dataset.shape[1:]

        step_indices = range(len(steps))
        if self._lead_time_hours is not None:
            step_indices = [
                i for i in step_indices
                if int(round(steps[i] / np.timedelta64(1, 'h'))) in self._lead_time_hours
            ]

        for ref_idx, base_date in enumerate(base_dates):
            for step_idx in step_indices:
                step = steps[step_idx]
                valid_time = base_date + step
                if self._split_start_date is not None and valid_time < self._split_start_date:
                    continue
                if self._split_end_date is not None and valid_time > self._split_end_date:
                    continue
                target_idx = np.nonzero(target_dates == valid_time)[0]
                assert (len(target_idx) != 0 or self.target_missing_as_zeros), \
                    f"Expected at least one target match for valid_time={valid_time} (base_date={base_date}, step={step}), found {len(target_idx)}."
                assert len(target_idx) <= 1, \
                    f"Expected no more than one target match for valid_time={valid_time} (base_date={base_date}, step={step}), found {len(target_idx)}."
                self._pairs.append((ref_idx, step_idx, None if len(target_idx) == 0 else target_idx[0]))

    def _input_frequency_hours(self, input_anemoi_dataset_path: str, input_frequency: str = None) -> int:
        """Effective input time step, in whole hours.

        Uses the explicit ``input_frequency`` override when given (which also drives anemoi's
        resampling), otherwise reads the native step from the dataset's timestamps.
        """
        if input_frequency is not None:
            hours = to_timedelta(input_frequency).total_seconds() / 3600
        else:
            steps = open_dataset(input_anemoi_dataset_path).steps
            hours = (steps[1] - steps[0]) / np.timedelta64(1, 'h')
        return int(round(hours))

    def __getitem__(self, idx):
        """Get input and target data for one (reference_time, step) pair.
        Transform and normalize, but do not interpolate."""

        ref_idx, step_idx, target_idx = self._pairs[idx]
        date_str = to_datetime(self._input_dataset.base_dates[ref_idx] + self._input_dataset.steps[step_idx]).strftime('%Y%m%d-%H%M')

        # Reorder the variable axis to input_channel_names order (the store's native
        # order isn't guaranteed to match - see _open_input_dataset), then squeeze
        # the ensemble dimension.
        input_data = self._input_dataset[ref_idx, :, :, step_idx, :][self._channel_indices].squeeze()

        # Pull target data at the same valid time as the input (squeeze the
        # ensemble dimension). target_idx is None only when target_missing_as_zeros
        # is set and the target has no data for this date, in which case a zeros
        # tensor is returned.
        if target_idx is None:
            target_data = np.zeros(self._target_sample_shape, dtype=np.float32).squeeze()
        else:
            target_data = self._output_dataset[target_idx].squeeze()

        # If target is COSMO, see commented code in base class's __getitem__ method,
        # regarding reshaping.

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

    def base_time(self) -> List:
        """Get reference/issue-time ('YYYYMMDD-HHMM') values, one per (reference_time, step)
        pair. Used to disambiguate pairs whose valid times overlap."""
        return [
            to_datetime(self._input_dataset.base_dates[ref_idx]).strftime('%Y%m%d-%H%M')
            for ref_idx, step_idx, _ in self._pairs
        ]

ANEMOI_IFSN320_REAL = AnemoiForecastDataset
ANEMOI_IFSO1280_REAL = AnemoiForecastDataset
