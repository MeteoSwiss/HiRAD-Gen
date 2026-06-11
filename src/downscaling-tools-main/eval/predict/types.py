"""Shared types for :mod:`eval.predict`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_EXTRA_ARGS_JSON = (
    '{"schedule_type":"experimental_piecewise","num_steps":30,'
    '"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,'
    '"high_schedule_type":"exponential","low_schedule_type":"karras",'
    '"num_steps_high":10,"num_steps_low":20,"rho":7.0,'
    '"sampler":"heun","S_churn":2.5,"S_min":0.75,'
    '"S_max":1000.0,"S_noise":1.05}'
)
DEFAULT_MEMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_STEPS = [24, 48, 72, 96, 120]
DEFAULT_DATES = ["20230826", "20230827", "20230828", "20230829", "20230830"]


@dataclass(frozen=True)
class BundleKey:
    """Unique identifier for a single input bundle."""

    date: str
    step: int
    member: int


@dataclass
class PredictionConfig:
    """Runtime configuration for modular prediction generation."""

    checkpoint_path: Path
    input_dir: Path
    output_dir: Path
    device: str = "cuda"
    precision: str = "fp32"
    num_gpus_per_model: int = 1
    validation_frequency: str = "50h"
    extra_args_json: str = DEFAULT_EXTRA_ARGS_JSON
    output_weather_state_mode: str = "surface-plus-core-pl"
    output_weather_states: list[str] | None = None
    slim_output: bool = True
    allow_overwrite: bool = False
    rank0_write_wait_seconds: int = 7200
    checkpoint_id: str = ""
    input_root_mode: str = "auto"
    allow_existing_out_dir: bool = False
    allow_missing_target_unsafe: bool = False
    manifest_path: Path | None = None
    members: list[int] = field(default_factory=lambda: list(DEFAULT_MEMBERS))
    steps: list[int] = field(default_factory=lambda: list(DEFAULT_STEPS))
    dates: list[str] = field(default_factory=lambda: list(DEFAULT_DATES))
    bundle_pairs: str = ""


@dataclass
class PredictionMetadata:
    """Provenance metadata attached to a predictions dataset."""

    init_date: np.datetime64
    lead_step_hours: int
    member_ids: list[int]
    checkpoint_id: str
    checkpoint_path: str
    source_bundle_paths: list[str]
    sampling_config_json: str
    validation_frequency: str
    output_weather_state_mode: str
    output_weather_states: list[str]
    target_members_with_data: int
    target_members_missing_data: int
    target_missing_member_ids: list[int]
    x_interp_exported: bool
    slim_output: bool
    missing_target_policy: str | None = None

    def to_attrs(self) -> dict[str, str | int]:
        """Return NetCDF-safe global attributes."""

        return {
            "init_date": np.datetime_as_string(self.init_date.astype("datetime64[ns]"), unit="s"),
            "lead_step_hours": int(self.lead_step_hours),
            "member_ids": ",".join(str(member) for member in self.member_ids),
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "sampling_config_json": self.sampling_config_json,
            "validation_frequency": self.validation_frequency,
            "target_members_with_data": int(self.target_members_with_data),
            "target_members_missing_data": int(self.target_members_missing_data),
            "target_missing_member_ids": ",".join(str(member) for member in self.target_missing_member_ids),
            "output_weather_state_mode": self.output_weather_state_mode,
            "output_weather_states": ",".join(self.output_weather_states),
            "target_weather_state_count": int(len(self.output_weather_states)),
            "x_interp_exported": int(bool(self.x_interp_exported)),
            "slim_output": int(bool(self.slim_output)),
            "source_bundle_count": int(len(self.source_bundle_paths)),
        }


@dataclass
class PredictionResult:
    """Prediction tensors and metadata for one bundle/member."""

    x: np.ndarray
    y: np.ndarray | None
    y_pred: np.ndarray
    lon_lres: np.ndarray
    lat_lres: np.ndarray
    lon_hres: np.ndarray
    lat_hres: np.ndarray
    weather_states: list[str]
    bundle_path: Path
    dates: np.ndarray | None = None


@dataclass
class EnsemblePrediction:
    """Stacked member predictions for one date/step output file."""

    init_date: np.datetime64
    lead_step_hours: int
    member_ids: list[int]
    source_bundle_paths: list[str]
    members_missing_target: list[int]
    weather_states: list[str]
    lon_lres: np.ndarray | None = None
    lat_lres: np.ndarray | None = None
    lon_hres: np.ndarray | None = None
    lat_hres: np.ndarray | None = None
    x_stack: np.ndarray | None = None
    y_stack: np.ndarray | None = None
    y_pred_stack: np.ndarray | None = None
    x_interp_stack: np.ndarray | None = None
    used_missing_target_unsafe: bool = False

    @property
    def target_members_with_data(self) -> int:
        """Return how many members contain target truth."""

        return max(0, len(self.member_ids) - len(self.members_missing_target))
