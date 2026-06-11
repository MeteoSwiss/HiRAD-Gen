"""Shared data structures for TC evaluation. Zero internal imports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SupportMode = Literal["native", "regridded"]
FORECAST_STEP_COUNT = 5


@dataclass(frozen=True)
class BoundingBox:
    """Spatial crop region. This is the EVALUATION box, not the data extent."""

    north: float
    south: float
    east: float
    west: float

    @property
    def crosses_dateline(self) -> bool:
        return self.east < self.west


@dataclass(frozen=True)
class CurveVectors:
    msl: np.ndarray
    wind: np.ndarray


@dataclass(frozen=True)
class StructuredGrid:
    lat_axis: np.ndarray
    lon_axis: np.ndarray
    point_indices: np.ndarray


def step_to_index(step: int) -> int:
    idx = (int(step) // 24) - 1
    if idx < 0 or idx > 4:
        raise ValueError(f"Unsupported step={step}; expected one of 24,48,72,96,120")
    return idx
