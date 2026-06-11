"""Inference-loop wrappers for bundle-based ensemble prediction."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from manual_inference.prediction.predict import _compute_x_interp_for_export, _predict_from_bundle

from .bundle_manager import date_str_to_datetime64
from .types import BundleKey, EnsemblePrediction, PredictionResult


def _squeeze_leading_singletons(array: np.ndarray, target_ndim: int) -> np.ndarray:
    squeezed = np.asarray(array)
    while squeezed.ndim > target_ndim and squeezed.shape[0] == 1:
        squeezed = squeezed[0]
    if squeezed.ndim != target_ndim:
        raise ValueError(
            f"Unsupported array shape {tuple(np.asarray(array).shape)}; expected {target_ndim} dimensions "
            f"after removing leading singleton axes."
        )
    return squeezed.astype(np.float32, copy=False)


def _cleanup_member_outputs(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_single_bundle(
    inference_model,
    datamodule,
    device: str,
    bundle_path: Path,
    *,
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "surface-plus-core-pl",
    output_weather_states: Sequence[str] | None = None,
) -> PredictionResult:
    """Run prediction for a single input bundle and return structured outputs."""

    x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, dates = _predict_from_bundle(
        inference_model=inference_model,
        datamodule=datamodule,
        device=device,
        bundle_nc=str(bundle_path),
        member_index=0,
        extra_args=extra_args,
        precision=precision,
        model_comm_group=model_comm_group,
        output_weather_state_mode=output_weather_state_mode,
        output_weather_states=output_weather_states,
    )
    return PredictionResult(
        x=np.asarray(x),
        y=None if y is None else np.asarray(y),
        y_pred=np.asarray(y_pred),
        lon_lres=np.asarray(lon_lres),
        lat_lres=np.asarray(lat_lres),
        lon_hres=np.asarray(lon_hres),
        lat_hres=np.asarray(lat_hres),
        weather_states=[str(name) for name in weather_states],
        bundle_path=bundle_path,
        dates=None if dates is None else np.asarray(dates),
    )


def predict_ensemble_members(
    inference_model,
    datamodule,
    device: str,
    bundle_map: dict[BundleKey, Path],
    date: str,
    step: int,
    members: Sequence[int],
    *,
    extra_args: dict,
    precision: str,
    model_comm_group,
    allow_missing_target_unsafe: bool = False,
    output_weather_state_mode: str = "surface-plus-core-pl",
    output_weather_states: Sequence[str] | None = None,
    keep_outputs: bool = True,
) -> EnsemblePrediction:
    """Predict all requested members for one date/step pair."""

    x_members: list[np.ndarray] = []
    y_members: list[np.ndarray | None] = []
    y_pred_members: list[np.ndarray] = []
    source_bundle_paths: list[str] = []
    members_missing_target: list[int] = []
    lon_lres = lat_lres = lon_hres = lat_hres = None
    weather_states: list[str] = []

    for member in members:
        bundle_path = bundle_map[BundleKey(date, step, int(member))]
        result = predict_single_bundle(
            inference_model,
            datamodule,
            device,
            bundle_path,
            extra_args=extra_args,
            precision=precision,
            model_comm_group=model_comm_group,
            output_weather_state_mode=output_weather_state_mode,
            output_weather_states=output_weather_states,
        )
        lon_lres = result.lon_lres
        lat_lres = result.lat_lres
        lon_hres = result.lon_hres
        lat_hres = result.lat_hres
        weather_states = result.weather_states
        source_bundle_paths.append(str(bundle_path))
        if result.y is None:
            members_missing_target.append(int(member))

        if keep_outputs:
            x_members.append(_squeeze_leading_singletons(result.x, 2))
            y_members.append(None if result.y is None else _squeeze_leading_singletons(result.y, 2))
            y_pred_members.append(_squeeze_leading_singletons(result.y_pred, 2))

        del result
        _cleanup_member_outputs(device)

    ensemble = EnsemblePrediction(
        init_date=date_str_to_datetime64(date),
        lead_step_hours=int(step),
        member_ids=[int(member) for member in members],
        source_bundle_paths=source_bundle_paths,
        members_missing_target=members_missing_target,
        weather_states=weather_states,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
    )
    if not keep_outputs:
        return ensemble

    x_stack = np.stack(x_members, axis=0)[None, ...]
    try:
        x_interp_stack = _compute_x_interp_for_export(
            inference_model=inference_model,
            x=x_stack,
            device=device,
            model_comm_group=model_comm_group,
        )
    except RuntimeError as exc:
        if "cannot export x_interp" not in str(exc):
            raise
        x_interp_stack = None
        print(
            f"WARNING date={date} step={step}: skipping x_interp export because the "
            f"inference model does not expose interpolation hooks ({exc})."
        )

    y_pred_stack = np.stack(y_pred_members, axis=0)[None, ...]
    used_missing_target_unsafe = False
    if any(member_y is not None for member_y in y_members):
        template = next(member_y for member_y in y_members if member_y is not None)
        y_filled = [
            member_y if member_y is not None else np.full_like(template, np.nan, dtype=np.float32)
            for member_y in y_members
        ]
        y_stack = np.stack(y_filled, axis=0)[None, ...]
        if members_missing_target:
            print(
                f"WARNING date={date} step={step}: missing target y for members {members_missing_target}; "
                "filled with NaN to keep y present in predictions file."
            )
    else:
        if not allow_missing_target_unsafe:
            raise SystemExit(
                f"No target truth (y) extracted for date={date} step={step}. "
                "Rebuild bundles with target_hres_* fields."
            )
        y_stack = np.full_like(y_pred_stack, np.nan, dtype=np.float32)
        used_missing_target_unsafe = True
        print(
            f"WARNING date={date} step={step}: no target y for any selected member; "
            "filled all y with NaN because --allow-missing-target-unsafe was set. "
            "Treat this file as prediction-only and non-canonical for truth-aware evaluation."
        )

    ensemble.x_stack = x_stack
    ensemble.y_stack = y_stack
    ensemble.y_pred_stack = y_pred_stack
    ensemble.x_interp_stack = x_interp_stack
    ensemble.used_missing_target_unsafe = used_missing_target_unsafe
    return ensemble
