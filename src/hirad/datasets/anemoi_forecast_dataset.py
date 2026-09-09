from .anemoi_dataset import AnemoiDataset

from anemoi.datasets import open_dataset
import datetime
import numpy as np
from pandas import to_datetime, to_timedelta
import torch
from typing import List

from hirad.utils.console import PythonLogger

logger = PythonLogger(__name__)


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
                provide_lead_time: bool = False,
                max_lead_hours: int = None,
                ):
        # Split by base (init) time: start_date/end_date are the first/last forecast init
        # times, and _align_input_output keeps only base times within that range. The target is
        # opened over exactly the valid-time window those inits produce: from the earliest
        # (start + smallest non-zero lead; lead 0 is never in the store) to the latest
        # (end + largest lead), clamped to the target's own availability. That way every lead of
        # every kept init has a matching target and nothing is dropped, except leads whose valid
        # time falls past the end of the target (handled in _align_input_output). A target field
        # near a split boundary may be used by both train and val (a train init reaching, via a
        # long lead, a valid time inside val); this small leakage is accepted by design.
        self._pair_start = to_datetime(start_date) if start_date is not None else None
        self._pair_end = to_datetime(end_date) if end_date is not None else None
        # Optional cap on the forecast lead used from the store (in hours); steps beyond it are
        # skipped in _align_input_output and excluded from the target window below.
        self._max_lead_hours = max_lead_hours
        target_start, target_end = start_date, end_date
        if start_date is not None and end_date is not None:
            steps = open_dataset(input_anemoi_dataset_path).steps
            nonzero = steps[steps > np.timedelta64(0, "h")]
            min_lead, max_lead = to_timedelta(nonzero.min()), to_timedelta(steps.max())
            if max_lead_hours is not None:
                max_lead = min(max_lead, to_timedelta(max_lead_hours, unit="h"))
            tgt_dates = open_dataset(target_anemoi_dataset_path).dates
            avail_start, avail_end = to_datetime(tgt_dates[0]), to_datetime(tgt_dates[-1])
            target_start = max(self._pair_start + min_lead, avail_start)
            target_end = min(self._pair_end + max_lead, avail_end)
        super().__init__(
            type=type,
            input_anemoi_dataset_path=input_anemoi_dataset_path,
            target_anemoi_dataset_path=target_anemoi_dataset_path,
            input_stats_anemoi_dataset_path=input_stats_anemoi_dataset_path,
            target_stats_anemoi_dataset_path=target_stats_anemoi_dataset_path,
            start_date=target_start,  # target valid-time window [start+min_lead, end+max_lead]
            end_date=target_end,
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
        # When True, __getitem__ appends an integer lead-time label (lead hours) used by
        # lead-time-conditioned models. The training/inference loops detect it via this
        # attribute (see TrainingManagerCorrDiff.load_and_preprocess_batch).
        self.provides_lead_time = provide_lead_time

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
        # The trajectory store can't be opened with select= (see _open_input_dataset), so its
        # native-order statistics must be reordered to input_channel_names order (via
        # self._channel_indices), same as the data are in __getitem__. Without this the
        # per-channel mean/std are mismatched to the (reordered) data channels.
        stats = self._input_dataset.statistics
        return {k: v[self._channel_indices] for k, v in stats.items()}

    def _align_input_output(self):
        """
        Align input and output samples by valid time.

        Iteration is driven by the input (forecast) dataset: each input
        (base_time, step) is mapped to the target index at the same
        valid_time (= base_time + step). Only base (init) times within
        [self._pair_start, self._pair_end] are kept (train/val split by init time).
        Each member of `_pairs` is a tuple (base_date_idx, step_idx, target_idx).
        A pair whose valid_time has no target is emitted with target_idx=None when
        `target_missing_as_zeros` (generation), otherwise dropped (a lead whose valid
        time falls past the end of the target's availability).
        """
        base_dates = to_datetime(self._input_dataset.base_dates)
        steps = self._input_dataset.steps

        # Note: assumes output dataset is a reanalysis dataset, not a forecast dataset.
        # O(1) valid-time -> target-index lookup (a per-pair scan is far too slow at the
        # hundreds of thousands of pairs a full training store produces).
        output_date_to_idx = {
            to_datetime(d): i for i, d in enumerate(self._output_dataset.dates)
        }

        self._pairs = []
        n_dropped = 0

        # Shape of a single (pre-squeeze) target sample, used to emit zeros when missing.
        self._target_sample_shape = self._output_dataset.shape[1:]

        for ref_idx, base_date in enumerate(base_dates):
            if self._pair_start is not None and base_date < self._pair_start:
                continue
            if self._pair_end is not None and base_date > self._pair_end:
                continue
            for step_idx, step in enumerate(steps):
                if self._max_lead_hours is not None and step / np.timedelta64(1, "h") > self._max_lead_hours:
                    continue
                valid_time = to_datetime(base_date + step)
                target_idx = output_date_to_idx.get(valid_time)
                if target_idx is None:
                    if self.target_missing_as_zeros:
                        self._pairs.append((ref_idx, step_idx, None))
                    else:
                        n_dropped += 1
                    continue
                self._pairs.append((ref_idx, step_idx, target_idx))

        if n_dropped:
            logger.warning(
                f"AnemoiForecastDataset: dropped {n_dropped} (base_date, step) pairs whose valid "
                f"time has no target (leads past the target range); {len(self._pairs)} pairs kept."
            )

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

        if self.provides_lead_time:
            lead_hours = int(round(self._input_dataset.steps[step_idx] / np.timedelta64(1, 'h')))
            return torch.from_numpy(target_data.copy()),\
                    torch.from_numpy(input_data),\
                    date_str,\
                    lead_hours

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
