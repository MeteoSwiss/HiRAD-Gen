from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_json_like(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _candidate_roots(path: Path, depth: int = 3) -> Iterable[Path]:
    current = path.resolve()
    for _ in range(depth + 1):
        yield current
        if current.parent == current:
            break
        current = current.parent


def find_experiment_config(pred_dir: Path) -> Path | None:
    for root in _candidate_roots(pred_dir):
        candidate = root / "EXPERIMENT_CONFIG.yaml"
        if candidate.exists():
            return candidate
    return None


def resolve_checkpoint_path(*, pred_dir: Path, ds: xr.Dataset | None = None, explicit_path: str = "") -> Path | None:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    candidates: list[str] = []
    if ds is not None:
        for key in ("checkpoint_path", "name_ckpt"):
            value = str(ds.attrs.get(key, "")).strip()
            if value:
                candidates.append(value)

    config_path = find_experiment_config(pred_dir)
    if config_path is not None:
        config = _load_json_like(config_path)
        checkpoint = config.get("checkpoint")
        if isinstance(checkpoint, dict):
            value = str(checkpoint.get("path", "")).strip()
            if value:
                candidates.append(value)

    for value in candidates:
        candidate = Path(value).expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


class CheckpointResidualInterpolator:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self._interface = None

    @property
    def inference_checkpoint_path(self) -> Path:
        if self.checkpoint_path.name.startswith("inference-"):
            return self.checkpoint_path
        return self.checkpoint_path.with_name(f"inference-{self.checkpoint_path.name}")

    def _load_interface(self):
        if self._interface is not None:
            return self._interface
        if torch is None:
            raise RuntimeError("torch is required to reconstruct x_interp from checkpoint interpolation.")
        inference_path = self.inference_checkpoint_path
        if not inference_path.exists():
            raise FileNotFoundError(f"Inference checkpoint not found for x_interp reconstruction: {inference_path}")
        interface = torch.load(inference_path, map_location=torch.device("cpu"), weights_only=False)
        if hasattr(interface, "eval"):
            interface.eval()
        self._interface = interface
        return interface

    def interpolate(self, x_values: np.ndarray) -> np.ndarray:
        interface = self._load_interface()
        model = getattr(interface, "model", interface)
        if not hasattr(model, "apply_interpolate_to_high_res"):
            raise RuntimeError(
                "Loaded inference checkpoint does not expose apply_interpolate_to_high_res; "
                f"checkpoint={self.inference_checkpoint_path}"
            )

        x_arr = np.asarray(x_values, dtype=np.float32)
        if x_arr.ndim == 2:
            x_tensor = torch.from_numpy(x_arr)[None, None, ...]
        elif x_arr.ndim == 3:
            x_tensor = torch.from_numpy(x_arr)[None, ...]
        else:
            raise ValueError(f"Unsupported low-resolution input shape for interpolation: {x_arr.shape}")

        with torch.inference_mode():
            try:
                interpolated = model.apply_interpolate_to_high_res(
                    x_tensor,
                    grid_shard_shapes=None,
                    model_comm_group=None,
                )
            except TypeError:
                interpolated = model.apply_interpolate_to_high_res(x_tensor)
        interpolated_np = interpolated.detach().cpu().numpy().astype(np.float64)

        if x_arr.ndim == 2:
            return interpolated_np[0, 0]
        return interpolated_np[0]

