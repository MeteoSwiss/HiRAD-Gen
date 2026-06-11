"""Dataclasses for scoreboard scoring results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RowClassification(Enum):
    """Classification of a TC stats row by its experiment label."""

    ANALYSIS = "analysis"
    REFERENCE = "reference"
    EEFO = "eefo"
    MODEL = "model"
    UNKNOWN = "unknown"


@dataclass
class TCScoreResult:
    """Result of TC extreme scoring for a single event."""

    event: str
    score: float | None = None
    enfo_deviation: float | None = None
    eefo_floor: float | None = None
    raw_score: float | None = None


@dataclass
class SpectraMetrics:
    """Per-field and mean spectra L2 distances."""

    fields: dict[str, float | None] = field(default_factory=dict)
    mean: float | None = None
    source_path: str = ""
    coverage: str = "missing"
    n_curves: int | None = None
    count_label: str = "pairs"
    counts: dict[str, int] = field(default_factory=dict)
    missing_reference_pairs: dict[str, int] = field(default_factory=dict)
    score_wavenumber_min_exclusive: float = 100.0


@dataclass
class SurfaceMetrics:
    """Surface loss metrics with per-variable breakdown."""

    weighted_mse: float | None = None
    weighted_nmse: float | None = None
    normalization_scheme: str = ""
    truth_std_by_variable: dict[str, float] = field(default_factory=dict)
    variables: dict[str, dict] = field(default_factory=dict)
    top_contributors: list[dict] = field(default_factory=list)
    source_path: str = ""


@dataclass
class ScoreRecord:
    """Single metric from an evaluator's score() output."""

    evaluator: str
    metric: str
    value: float
    unit: str
