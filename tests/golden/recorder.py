"""Storage for golden records.

A record is a named value captured while running a real entry point. Three kinds:
  - ``array``:   a full numpy array, stored as ``.npy`` (compared element-wise).
  - ``summary``: a fingerprint of a tensor too large to store (per-channel statistics,
                 a SHA-256 of the logical float32 bytes and values at fixed sampled positions).
  - ``json``:    a JSON-serialisable value (compared for equality).
Record names are semantic (e.g. ``step1/inputs/image_lr``) so that references stay valid while
the code that produces them is refactored.
"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

N_SAMPLED_VALUES = 8192


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().contiguous().numpy()
    return np.asarray(value)


def tensor_summary(value) -> dict:
    """Layout-independent fingerprint of a (B, C, ...) tensor."""
    array = to_numpy(value)
    as64 = array.astype(np.float64)
    channel_axis = 1 if array.ndim >= 2 else 0
    reduce_axes = tuple(i for i in range(array.ndim) if i != channel_axis)
    rng = np.random.default_rng(0)
    flat = as64.reshape(-1)
    positions = rng.choice(flat.size, size=min(N_SAMPLED_VALUES, flat.size), replace=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array.astype(np.float32)).tobytes()).hexdigest(),
        "channel_mean": as64.mean(axis=reduce_axes).tolist(),
        "channel_std": as64.std(axis=reduce_axes).tolist(),
        "channel_min": as64.min(axis=reduce_axes).tolist(),
        "channel_max": as64.max(axis=reduce_axes).tolist(),
        "channel_abs_mean": np.abs(as64).mean(axis=reduce_axes).tolist(),
        "sampled_positions": positions.tolist(),
        "sampled_values": flat[positions].tolist(),
    }


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.records_dir = self.out_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, dict] = {}
        self.step = 0
        self._context: list[str] = []

    @property
    def context(self) -> str:
        return self._context[-1] if self._context else "none"

    @contextmanager
    def in_context(self, name: str):
        self._context.append(name)
        try:
            yield
        finally:
            self._context.pop()

    def key(self, name: str) -> str:
        return f"step{self.step}/{name}" if self.step else name

    def has(self, name: str) -> bool:
        return self.key(name) in self.index

    def array(self, name: str, value) -> None:
        key = self.key(name)
        if value is None:
            self.index[key] = {"kind": "json", "value": None}
            return
        array = to_numpy(value)
        file_name = key.replace("/", "__") + ".npy"
        np.save(self.records_dir / file_name, array)
        self.index[key] = {"kind": "array", "file": file_name, "dtype": str(array.dtype), "shape": list(array.shape)}

    def summary(self, name: str, value) -> None:
        self.index[self.key(name)] = {"kind": "summary", "value": tensor_summary(value)}

    def json(self, name: str, value) -> None:
        self.index[self.key(name)] = {"kind": "json", "value": value}

    def save(self, manifest: dict) -> None:
        with open(self.out_dir / "records.json", "w") as f:
            json.dump(self.index, f, indent=1)
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=1, default=str)


def load_records(out_dir: Path) -> dict:
    with open(Path(out_dir) / "records.json") as f:
        return json.load(f)
