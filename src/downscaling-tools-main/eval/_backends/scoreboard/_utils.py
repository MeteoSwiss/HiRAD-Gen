"""Shared helpers used across eval.scoreboard.* modules.

Single home for guard-and-cast and JSON read primitives that were duplicated
across tc.py, spectra.py, surface.py, and eval/jobs/scoreboard_metrics.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def finite_float(value: Any) -> float | None:
    """Convert *value* to float; return None if non-numeric or non-finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file; return its top-level dict, or {} if not a dict."""
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}
