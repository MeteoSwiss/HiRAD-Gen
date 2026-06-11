"""Rank-0 synchronized file-writing helpers for distributed prediction jobs."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import xarray as xr

RANK0_WRITE_DONE_SUFFIX = ".rank0-write-done"
RANK0_WRITE_FAILED_SUFFIX = ".rank0-write-failed"


def _distributed_barrier(*, args_device: str, local_rank: int, world_size: int) -> None:
    """Mirror the legacy distributed barrier helper."""

    if world_size <= 1 or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    sync_device_ids = [local_rank] if args_device == "cuda" and torch.cuda.is_available() else None
    torch.distributed.barrier(device_ids=sync_device_ids)


def _destroy_process_group() -> None:
    """Destroy the default distributed process group if it exists."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.destroy_process_group()


class Rank0FileWriter:
    """Rank 0 writes a file and marks completion; other ranks wait on marker files."""

    def __init__(self, global_rank: int, timeout_seconds: int = 7200, poll_seconds: float = 5.0):
        self.global_rank = int(global_rank)
        self.timeout_seconds = int(timeout_seconds)
        self.poll_seconds = float(poll_seconds)

    def _done_marker(self, path: Path) -> Path:
        return path.with_name(path.name + RANK0_WRITE_DONE_SUFFIX)

    def _failed_marker(self, path: Path) -> Path:
        return path.with_name(path.name + RANK0_WRITE_FAILED_SUFFIX)

    def _clear_markers(self, path: Path) -> None:
        """Remove stale done/failed markers for ``path``."""

        for marker in (self._done_marker(path), self._failed_marker(path)):
            if marker.exists():
                marker.unlink()

    def _mark_complete(self, path: Path) -> None:
        """Write the success marker for ``path``."""

        self._done_marker(path).write_text("ok\n", encoding="utf-8")

    def _mark_failed(self, path: Path, exc: BaseException) -> None:
        """Write the failure marker for ``path``."""

        self._failed_marker(path).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    def _wait_for_complete(self, path: Path) -> None:
        """Wait until rank 0 writes either a success or failure marker."""

        if self.global_rank == 0:
            return
        done_marker = self._done_marker(path)
        failed_marker = self._failed_marker(path)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if done_marker.exists():
                return
            if failed_marker.exists():
                detail = failed_marker.read_text(encoding="utf-8").strip()
                raise SystemExit(f"Rank-0 write failed for {path}: {detail or 'unknown error'}")
            time.sleep(self.poll_seconds)
        raise TimeoutError(
            f"Timed out waiting for rank 0 to finish writing {path} after {self.timeout_seconds}s."
        )

    def write_or_wait(self, ds: xr.Dataset | None, out_path: Path) -> None:
        """Rank 0 writes ``ds`` and other ranks wait for completion."""

        if self.global_rank != 0:
            self._wait_for_complete(out_path)
            return
        if ds is None:
            raise ValueError("Rank 0 requires a dataset to write.")
        self._clear_markers(out_path)
        try:
            ds.to_netcdf(out_path)
        except BaseException as exc:
            self._mark_failed(out_path, exc)
            raise
        self._mark_complete(out_path)
