"""Test helpers for :mod:`eval.predict`."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr

PREDICT_DIR = Path(__file__).resolve().parents[1]


def _drop_module_tree(monkeypatch, root: str) -> None:
    for name in [name for name in sys.modules if name == root or name.startswith(f"{root}.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def install_predict_package(monkeypatch) -> None:
    """Register a lightweight ``eval.predict`` package without running its ``__init__``."""

    import eval as eval_pkg

    _drop_module_tree(monkeypatch, "eval.predict")
    pkg = types.ModuleType("eval.predict")
    pkg.__file__ = str(PREDICT_DIR / "__init__.py")
    pkg.__package__ = "eval.predict"
    pkg.__path__ = [str(PREDICT_DIR)]
    spec = importlib.machinery.ModuleSpec("eval.predict", loader=None, is_package=True)
    spec.submodule_search_locations = [str(PREDICT_DIR)]
    pkg.__spec__ = spec
    monkeypatch.setitem(sys.modules, "eval.predict", pkg)
    monkeypatch.setattr(eval_pkg, "predict", pkg, raising=False)


def stub_build_predictions_dataset(
    *,
    x,
    y,
    y_pred,
    lon_lres,
    lat_lres,
    lon_hres,
    lat_hres,
    weather_states,
    dates,
    member_ids,
    x_interp=None,
    include_member_views=True,
):
    """Build a minimal predictions dataset for unit tests."""

    del include_member_views
    x = np.asarray(x)
    y_pred = np.asarray(y_pred)
    lon_lres = np.asarray(lon_lres)
    lat_lres = np.asarray(lat_lres)
    lon_hres = np.asarray(lon_hres)
    lat_hres = np.asarray(lat_hres)
    dates = np.asarray(dates if dates is not None else np.arange(x.shape[0]))

    ds = xr.Dataset(
        {
            "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), x),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y_pred),
            "date": (("sample",), dates),
            "lon_lres": (("grid_point_lres",), lon_lres),
            "lat_lres": (("grid_point_lres",), lat_lres),
            "lon_hres": (("grid_point_hres",), lon_hres),
            "lat_hres": (("grid_point_hres",), lat_hres),
        },
        coords={
            "sample": list(range(x.shape[0])),
            "ensemble_member": list(member_ids),
            "grid_point_lres": list(range(lon_lres.shape[0])),
            "grid_point_hres": list(range(lon_hres.shape[0])),
            "weather_state": list(weather_states),
        },
    )
    if y is not None:
        ds["y"] = (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.asarray(y))
    if x_interp is not None:
        ds["x_interp"] = (
            ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
            np.asarray(x_interp),
        )
    return ds


def install_manual_inference_dataset_stub(
    monkeypatch,
    build_fn: Callable[..., xr.Dataset] | None = None,
) -> None:
    """Install a light ``manual_inference.prediction.dataset`` stub."""

    _drop_module_tree(monkeypatch, "manual_inference")
    manual_pkg = types.ModuleType("manual_inference")
    manual_pkg.__package__ = "manual_inference"
    manual_pkg.__path__ = []

    prediction_pkg = types.ModuleType("manual_inference.prediction")
    prediction_pkg.__package__ = "manual_inference.prediction"
    prediction_pkg.__path__ = []

    dataset_mod = types.ModuleType("manual_inference.prediction.dataset")
    dataset_mod.__package__ = "manual_inference.prediction"
    dataset_mod.build_predictions_dataset = build_fn or stub_build_predictions_dataset
    dataset_mod.OUTPUT_WEATHER_STATE_MODE_CHOICES = ("all", "surface-plus-core-pl")

    manual_pkg.prediction = prediction_pkg
    prediction_pkg.dataset = dataset_mod

    monkeypatch.setitem(sys.modules, "manual_inference", manual_pkg)
    monkeypatch.setitem(sys.modules, "manual_inference.prediction", prediction_pkg)
    monkeypatch.setitem(sys.modules, "manual_inference.prediction.dataset", dataset_mod)


def install_torch_stub(
    monkeypatch,
    *,
    cuda_available: bool = False,
    distributed_available: bool = True,
    distributed_initialized: bool = False,
) -> None:
    """Install a tiny ``torch`` stub sufficient for distributed I/O tests."""

    _drop_module_tree(monkeypatch, "torch")

    torch_mod = types.ModuleType("torch")
    cuda_mod = types.ModuleType("torch.cuda")
    distributed_mod = types.ModuleType("torch.distributed")

    cuda_mod.is_available = lambda: bool(cuda_available)
    cuda_mod.empty_cache = lambda: None

    distributed_mod.is_available = lambda: bool(distributed_available)
    distributed_mod.is_initialized = lambda: bool(distributed_initialized)
    distributed_mod.barrier = lambda device_ids=None: None
    distributed_mod.destroy_process_group = lambda: None

    torch_mod.cuda = cuda_mod
    torch_mod.distributed = distributed_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed_mod)


def load_predict_submodule(
    monkeypatch,
    module_name: str,
    *,
    build_predictions_dataset_fn: Callable[..., xr.Dataset] | None = None,
):
    """Import a predict submodule with lightweight dependency stubs."""

    install_predict_package(monkeypatch)
    if module_name == "dataset_builder" or build_predictions_dataset_fn is not None:
        install_manual_inference_dataset_stub(monkeypatch, build_predictions_dataset_fn)
    if module_name == "distributed_io":
        install_torch_stub(monkeypatch)
    importlib.invalidate_caches()
    return importlib.import_module(f"eval.predict.{module_name}")
