"""Checkpoint identification and lane inference."""

from __future__ import annotations

import re
from pathlib import Path

# Resolution pair to lane mapping — from eval/jobs/checkpoint_profile.py
LANE_BY_RESOLUTION_PAIR = {
    (48, 96): "o48_o96",
    (96, 320): "o96_o320",
    (320, 1280): "o320_o1280",
    (1280, 2560): "o1280_o2560",
}

# Resolution extraction patterns — from eval/jobs/checkpoint_profile.py
_RESOLUTION_PATTERNS = (
    re.compile(r"mars-o(?P<res>\d+)"),
    re.compile(r"downscaling_od_o(?P<res>\d+)"),
    re.compile(r"(?:^|[/_-])o(?P<res>\d+)(?:[._/-]|$)"),
)

# Checkpoint ID extraction — search all path parts for hex strings
_CKPT_ID_RE = re.compile(r"([0-9a-f]{8,40})")

# Lightweight resolution extraction from individual path segments.
# Uses a lookahead-based pattern so adjacent markers like "o96_o320"
# both match even though they share a separator character.
_SEGMENT_RES_RE = re.compile(r"(?:^|(?<=[/_.-]))o(\d+)(?=[/_.-]|$)")


def identify_checkpoint(path: Path) -> dict:
    """Extract checkpoint metadata: id, path.

    Returns dict with keys: id, path, name.
    The id is the first hex string (8-40 chars) found in any path component.
    """
    path = Path(path)
    for part in reversed(path.parts):
        match = _CKPT_ID_RE.search(part)
        if match:
            return {"id": match.group(1), "path": str(path), "name": path.stem}
    return {"id": path.stem, "path": str(path), "name": path.stem}


def infer_lane(checkpoint_path: Path) -> str:
    """Infer lane name from checkpoint path components.

    Scans the path string for resolution markers (e.g. o96, o320) and maps
    the (lres, hres) pair to a lane name. This is a lightweight heuristic
    that works on the path alone — for authoritative lane inference from
    checkpoint metadata, use eval.jobs.checkpoint_profile.resolve_profile().
    """
    path_str = str(checkpoint_path)
    resolutions: list[int] = []
    seen: set[int] = set()

    # First try the specific named patterns
    for pattern in _RESOLUTION_PATTERNS[:2]:
        for match in pattern.finditer(path_str):
            res = int(match.group("res"))
            if res not in seen:
                seen.add(res)
                resolutions.append(res)

    # Then scan each path segment with the generic pattern (lookahead-based
    # so adjacent markers like "o96_o320" both match)
    for part in Path(checkpoint_path).parts:
        for match in _SEGMENT_RES_RE.finditer(part):
            res = int(match.group(1))
            if res not in seen:
                seen.add(res)
                resolutions.append(res)

    if len(resolutions) >= 2:
        resolutions_sorted = sorted(resolutions)
        for i in range(len(resolutions_sorted) - 1):
            pair = (resolutions_sorted[i], resolutions_sorted[i + 1])
            if pair in LANE_BY_RESOLUTION_PAIR:
                return LANE_BY_RESOLUTION_PAIR[pair]

    if len(resolutions) == 1:
        res = resolutions[0]
        for (lres, hres), lane in LANE_BY_RESOLUTION_PAIR.items():
            if res == lres or res == hres:
                return lane

    raise ValueError(
        f"Cannot infer lane from checkpoint path: {checkpoint_path}. "
        f"Found resolutions: {resolutions}. "
        f"Known pairs: {sorted(LANE_BY_RESOLUTION_PAIR)}"
    )
