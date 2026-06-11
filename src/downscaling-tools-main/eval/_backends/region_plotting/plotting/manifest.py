"""Shared manifest-writing helpers for region plotting scripts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_manifest(
    *,
    out_root: Path,
    payload: Mapping[str, Any],
    filename: str = "manifest.json",
) -> Path:
    """Write a JSON manifest under *out_root* and return its path."""
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / filename
    manifest_path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return manifest_path
