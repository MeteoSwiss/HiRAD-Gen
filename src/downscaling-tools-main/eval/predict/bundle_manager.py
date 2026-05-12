"""Bundle discovery and date/step parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import BundleKey

BUNDLE_RE = re.compile(
    r"date(?P<date>\d{8})_time\d{4}_mem(?P<member>\d+)_step(?P<step>\d{3})h_input_bundle\.nc$"
)


def parse_bundle_key(path: Path) -> BundleKey | None:
    """Parse a bundle filename into a :class:`BundleKey`."""

    match = BUNDLE_RE.search(path.name)
    if not match:
        return None
    return BundleKey(
        date=match.group("date"),
        step=int(match.group("step")),
        member=int(match.group("member")),
    )


def discover_bundles(input_root: Path, recursive: bool = True) -> dict[BundleKey, Path]:
    """Discover bundle files, keeping the newest path for duplicate keys."""

    bundle_map: dict[BundleKey, Path] = {}
    candidates = input_root.rglob("*_input_bundle.nc") if recursive else input_root.glob("*_input_bundle.nc")
    for path in sorted(candidates):
        key = parse_bundle_key(path)
        if key is None:
            continue
        if key not in bundle_map or path.stat().st_mtime >= bundle_map[key].stat().st_mtime:
            bundle_map[key] = path
    return bundle_map


def resolve_date_step_pairs(
    bundle_map: dict[BundleKey, Path],
    dates: Iterable[str],
    steps: Iterable[int],
    bundle_pairs_str: str,
) -> list[tuple[str, int]]:
    """Resolve requested date/step pairs from CLI-style inputs."""

    del bundle_map  # kept in signature for symmetry with the legacy script.
    pairs: list[tuple[str, int]] = []
    if bundle_pairs_str:
        for token in bundle_pairs_str.split(","):
            token = token.strip()
            if not token:
                continue
            date_str, step_str = token.split(":", maxsplit=1)
            pairs.append((date_str.strip(), int(step_str.strip())))
    else:
        pairs.extend((date_str, int(step)) for date_str in dates for step in steps)

    deduped: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        deduped.append(pair)
        seen.add(pair)
    return deduped


def date_str_to_datetime64(date_str: str) -> np.datetime64:
    """Convert ``YYYYMMDD`` into ``datetime64[ns]``."""

    return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00", "ns")
