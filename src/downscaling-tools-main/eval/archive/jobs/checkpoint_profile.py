#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from manual_inference.prediction.predict import _resolve_ckpt_path

LANE_BY_RESOLUTION_PAIR = {
    (48, 96): "o48_o96",
    (96, 320): "o96_o320",
    (320, 1280): "o320_o1280",
    (1280, 2560): "o1280_o2560",
}
VALID_SOURCE_HPC = {"ac", "ag", "leonardo", "jupiter"}
VALID_STACK = {"new", "old"}
_RESOLUTION_PATTERNS = (
    re.compile(r"mars-o(?P<res>\d+)"),
    re.compile(r"downscaling_od_o(?P<res>\d+)"),
    re.compile(r"(?:^|[/_-])o(?P<res>\d+)(?:[._/-]|$)"),
)


@dataclass(frozen=True)
class CheckpointProfile:
    checkpoint_path: str
    lane: str
    stack_flavor: str
    source_hpc: str
    host_short: str
    host_family: str
    recommended_venv: str


def _normalize_cfg(cfg: Any) -> Any:
    if hasattr(cfg, "model_dump"):
        cfg = cfg.model_dump()
    try:
        from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    return cfg


def _load_config_from_anemoi_metadata(checkpoint_path: str) -> Any:
    with zipfile.ZipFile(checkpoint_path) as zf:
        metadata_members = [
            name for name in zf.namelist() if name.endswith("anemoi-metadata/anemoi.json")
        ]
        if not metadata_members:
            raise RuntimeError(
                f"Checkpoint missing anemoi metadata JSON: {checkpoint_path}"
            )
        payload = json.loads(zf.read(metadata_members[0]))
    if not isinstance(payload, dict) or "config" not in payload:
        raise RuntimeError(
            f"Checkpoint metadata JSON missing config block: {checkpoint_path}"
        )
    return _normalize_cfg(payload["config"])


def _load_checkpoint_config(
    checkpoint_path: str,
    *,
    prefer_anemoi_metadata: bool = False,
) -> Any:
    if prefer_anemoi_metadata:
        return _load_config_from_anemoi_metadata(checkpoint_path)

    import torch  # pylint: disable=import-outside-toplevel

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {})
    if "config" not in hp:
        raise RuntimeError(
            f"Checkpoint missing hyper_parameters.config: {checkpoint_path}"
        )
    return _normalize_cfg(hp["config"])


def _host_family(host_short: str) -> str:
    if host_short.startswith("ac"):
        return "ac"
    if host_short.startswith("ag"):
        return "ag"
    raise ValueError(
        f"Unsupported host family for host={host_short!r}. Expected ac* or ag*."
    )


def _recommended_venv(stack_flavor: str, host_family: str) -> str:
    if stack_flavor not in VALID_STACK:
        raise ValueError(f"Invalid stack_flavor={stack_flavor!r}")
    if host_family == "ac":
        return (
            "/home/ecm5702/dev/.ds-dyn/bin/activate"
            if stack_flavor == "new"
            else "/home/ecm5702/dev/.ds-old/bin/activate"
        )
    if host_family == "ag":
        return (
            "/home/ecm5702/dev/.ds-ag/bin/activate"
            if stack_flavor == "new"
            else "/home/ecm5702/dev/.ds-ag-old/bin/activate"
        )
    raise ValueError(f"Unsupported host_family={host_family!r}")


def _iter_named_datasets(cfg: dict[str, Any], split: str) -> list[tuple[str, str]]:
    try:
        zipped = cfg["dataloader"][split]["dataset"]["zip"]
    except (KeyError, TypeError):
        # Multi-dataset config uses dataloader.<split>.datasets.<role>.dataset
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
            # Normalise role names: in_lres->lres, in_hres/out_hres->hres
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


def infer_stack_from_config(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, default=str)
    has_new_marker = "multi_dataset_normalizer" in blob or "DiffusionDownscaler" in blob
    has_old_marker = ".preprocessing.normalizer.TopNormalizer" in blob
    if has_new_marker and has_old_marker:
        raise RuntimeError("Ambiguous stack markers found in checkpoint config.")
    if has_new_marker:
        return "new"
    if has_old_marker:
        return "old"
    raise RuntimeError(
        "Could not infer stack flavor from checkpoint config markers."
    )


def resolve_profile(
    *,
    checkpoint_path: str,
    source_hpc: str,
    host_short: str,
    allow_inference_companion: bool = False,
    expected_lane: str | None = None,
    expected_stack_flavor: str | None = None,
    expected_venv: str | None = None,
) -> CheckpointProfile:
    if source_hpc not in VALID_SOURCE_HPC:
        raise ValueError(
            f"Invalid source_hpc={source_hpc!r}. Allowed: {sorted(VALID_SOURCE_HPC)}"
        )
    cfg = _load_checkpoint_config(
        checkpoint_path,
        prefer_anemoi_metadata=allow_inference_companion
        and os.path.basename(checkpoint_path).startswith("inference-"),
    )
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"Unsupported normalized config type {type(cfg)} from checkpoint."
        )

    lane = infer_lane_from_config(cfg)
    stack_flavor = infer_stack_from_config(cfg)
    host_family = _host_family(host_short)
    venv = _recommended_venv(stack_flavor, host_family)

    if expected_lane and expected_lane != lane:
        raise RuntimeError(
            f"Lane mismatch for checkpoint {checkpoint_path}: expected={expected_lane}, inferred={lane}"
        )
    if expected_stack_flavor and expected_stack_flavor != stack_flavor:
        raise RuntimeError(
            f"Stack mismatch for checkpoint {checkpoint_path}: expected={expected_stack_flavor}, inferred={stack_flavor}"
        )
    if expected_venv:
        expected_venv_norm = os.path.expanduser(expected_venv)
        venv_norm = os.path.expanduser(venv)
        if expected_venv_norm != venv_norm:
            raise RuntimeError(
                f"Venv mismatch for checkpoint {checkpoint_path}: "
                f"template={expected_venv_norm}, inferred={venv_norm}"
            )

    return CheckpointProfile(
        checkpoint_path=checkpoint_path,
        lane=lane,
        stack_flavor=stack_flavor,
        source_hpc=source_hpc,
        host_short=host_short,
        host_family=host_family,
        recommended_venv=venv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer strict checkpoint profile (lane, stack_flavor, recommended venv) "
            "and fail on mismatches."
        )
    )
    parser.add_argument("--name-ckpt", required=True)
    parser.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    parser.add_argument("--source-hpc", required=True, choices=sorted(VALID_SOURCE_HPC))
    parser.add_argument("--host-short", default=socket.gethostname().split(".")[0])
    parser.add_argument("--expected-lane", default="")
    parser.add_argument("--expected-stack-flavor", default="")
    parser.add_argument("--expected-venv", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-inference-companion",
        action="store_true",
        help=(
            "Allow profiling an inference-*.ckpt companion directly. "
            "Use this only for low-memory preflight when the full training checkpoint "
            "is too large to inspect safely on the current login node."
        ),
    )
    args = parser.parse_args()

    ckpt_path = _resolve_ckpt_path(
        args.name_ckpt,
        args.ckpt_root,
        allow_inference_companion=args.allow_inference_companion,
    )
    profile = resolve_profile(
        checkpoint_path=ckpt_path,
        source_hpc=args.source_hpc,
        host_short=args.host_short,
        allow_inference_companion=args.allow_inference_companion,
        expected_lane=args.expected_lane or None,
        expected_stack_flavor=args.expected_stack_flavor or None,
        expected_venv=args.expected_venv or None,
    )
    payload = asdict(profile)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for k, v in payload.items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
