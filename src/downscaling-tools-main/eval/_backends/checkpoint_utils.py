"""Utilities for inferring lane and other metadata from checkpoint configs.

Extracted from eval/archive/jobs/checkpoint_profile.py so that active
backends (e.g. sigma_evaluator) no longer depend on the archived module.
"""
from __future__ import annotations

import re
from typing import Any

LANE_BY_RESOLUTION_PAIR: dict[tuple[int, int], str] = {
    (48, 96): "o48_o96",
    (96, 320): "o96_o320",
    (320, 1280): "o320_o1280",
    (1280, 2560): "o1280_o2560",
}

_RESOLUTION_PATTERNS = (
    re.compile(r"mars-o(?P<res>\d+)"),
    re.compile(r"downscaling_od_o(?P<res>\d+)"),
    re.compile(r"(?:^|[/_-])o(?P<res>\d+)(?:[._/-]|$)"),
)


def _iter_named_datasets(cfg: dict[str, Any], split: str) -> list[tuple[str, str]]:
    try:
        zipped = cfg["dataloader"][split]["dataset"]["zip"]
    except (KeyError, TypeError):
        try:
            datasets_dict = cfg["dataloader"][split]["datasets"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot read dataloader.{split}.dataset.zip from checkpoint config."
            ) from exc
        out: list[tuple[str, str]] = []
        for role, ds_cfg in (datasets_dict.items() if hasattr(datasets_dict, "items") else []):
            path = ds_cfg.get("dataset") if isinstance(ds_cfg, dict) else None
            if path is None:
                continue
            if "lres" in role:
                name = "lres"
            elif "hres" in role or "out" in role:
                name = "hres"
            else:
                name = role
            if isinstance(path, list):
                for p in path:
                    out.append((name, str(p)))
            else:
                out.append((name, str(path)))
        return out

    out = []
    for item in zipped:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        dataset_value = item.get("dataset")
        if not name or dataset_value is None:
            continue
        if isinstance(dataset_value, list):
            candidates = [str(v) for v in dataset_value]
        else:
            candidates = [str(dataset_value)]
        for c in candidates:
            out.append((name, c))
    return out


def _extract_single_resolution(paths: list[str], role: str) -> int:
    resolutions = set()
    for p in paths:
        for pattern in _RESOLUTION_PATTERNS:
            m = pattern.search(p)
            if m:
                resolutions.add(int(m.group("res")))
                break
    if not resolutions:
        raise RuntimeError(f"Could not infer {role} resolution from dataset paths: {paths}")
    if len(resolutions) != 1:
        raise RuntimeError(
            f"Ambiguous {role} resolutions {sorted(resolutions)} from dataset paths: {paths}"
        )
    return next(iter(resolutions))


def infer_lane_from_config(cfg: dict[str, Any]) -> str:
    """Infer the lane name (e.g. 'o48_o96') from a checkpoint training config."""
    candidates = _iter_named_datasets(cfg, "validation")
    if not candidates:
        candidates = _iter_named_datasets(cfg, "training")

    paths_by_name: dict[str, list[str]] = {}
    for name, path in candidates:
        paths_by_name.setdefault(name, []).append(path)

    if "lres" not in paths_by_name:
        raise RuntimeError("Cannot infer lane: missing lres dataset in config.")
    if "hres" not in paths_by_name and "out" not in paths_by_name:
        raise RuntimeError("Cannot infer lane: missing hres/out datasets in config.")

    lres = _extract_single_resolution(paths_by_name["lres"], "lres")
    if "hres" in paths_by_name:
        hres = _extract_single_resolution(paths_by_name["hres"], "hres")
    else:
        hres = _extract_single_resolution(paths_by_name["out"], "out")

    pair = (lres, hres)
    if pair not in LANE_BY_RESOLUTION_PAIR:
        raise RuntimeError(
            f"Unsupported lane resolution pair {pair}. Known pairs: {sorted(LANE_BY_RESOLUTION_PAIR)}"
        )
    return LANE_BY_RESOLUTION_PAIR[pair]
