"""Canonical analysis data loader — replaces hardcoded dicts with YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"
_CANONICAL_PATH = _DATA_DIR / "canonical_analysis.yaml"
_CANONICAL_EEFO_PATH = _DATA_DIR / "canonical_eefo.yaml"

_cache: dict[str, Any] | None = None
_eefo_cache: dict[str, Any] | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_canonical_analysis(
    resolution: str | None = None,
    *,
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load canonical analysis data from YAML.

    Parameters
    ----------
    resolution : str, optional
        Resolution tag (e.g. "o320"). If None, returns the entire file.
    path : Path, optional
        Override path to YAML file. Defaults to bundled data file.

    Returns
    -------
    dict mapping event names to their canonical tail percentile dicts.
    If resolution is None, returns the full resolution→event→values mapping.
    """
    global _cache
    yaml_path = path or _CANONICAL_PATH
    if _cache is None or path is not None:
        loaded = _load_yaml(yaml_path)
        if path is None:
            _cache = loaded
    else:
        loaded = _cache

    if resolution is None:
        return loaded

    section = loaded.get(resolution)
    return dict(section) if isinstance(section, dict) else {}


def load_canonical_eefo(
    resolution: str | None = None,
    *,
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load canonical EEFO data from YAML.

    Parameters
    ----------
    resolution : str, optional
        Target resolution tag (e.g. "o1280"). If None, returns the entire file.
    path : Path, optional
        Override path to YAML file. Defaults to bundled data file.

    Returns
    -------
    dict mapping event names to their canonical EEFO tail percentile dicts.
    If resolution is None, returns the full resolution->event->values mapping.
    """
    global _eefo_cache
    yaml_path = path or _CANONICAL_EEFO_PATH
    if _eefo_cache is None or path is not None:
        loaded = _load_yaml(yaml_path)
        if path is None:
            _eefo_cache = loaded
    else:
        loaded = _eefo_cache

    if resolution is None:
        return loaded

    section = loaded.get(resolution)
    return dict(section) if isinstance(section, dict) else {}
