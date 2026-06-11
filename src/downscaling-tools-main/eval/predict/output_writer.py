"""Prediction dataset serialization and validation."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from .dataset_builder import build_prediction_dataset, validate_predictions_dataset
from .distributed_io import Rank0FileWriter
from .types import EnsemblePrediction, PredictionConfig, PredictionMetadata


def prediction_output_path(output_dir: Path, date: str, step: int) -> Path:
    """Return the canonical predictions filename for a date/step pair."""

    return output_dir / f"predictions_{date}_step{int(step):03d}.nc"


def _metadata_from_ensemble(
    ensemble: EnsemblePrediction,
    config: PredictionConfig,
) -> PredictionMetadata:
    checkpoint_id = config.checkpoint_id or config.checkpoint_path.parent.name or config.checkpoint_path.stem
    output_weather_states = [str(name) for name in ensemble.weather_states]
    return PredictionMetadata(
        init_date=ensemble.init_date,
        lead_step_hours=ensemble.lead_step_hours,
        member_ids=list(ensemble.member_ids),
        checkpoint_id=checkpoint_id,
        checkpoint_path=str(config.checkpoint_path),
        source_bundle_paths=list(ensemble.source_bundle_paths),
        sampling_config_json=config.extra_args_json,
        validation_frequency=config.validation_frequency,
        output_weather_state_mode=config.output_weather_state_mode,
        output_weather_states=output_weather_states,
        target_members_with_data=ensemble.target_members_with_data,
        target_members_missing_data=len(ensemble.members_missing_target),
        target_missing_member_ids=list(ensemble.members_missing_target),
        x_interp_exported=ensemble.x_interp_stack is not None,
        slim_output=bool(config.slim_output),
        missing_target_policy=(
            "all_nan_due_to_allow_missing_target_unsafe" if ensemble.used_missing_target_unsafe else None
        ),
    )


def write_predictions_file(
    ensemble: EnsemblePrediction,
    out_path: Path,
    config: PredictionConfig,
    writer: Rank0FileWriter,
) -> Path:
    """Build, validate, and write the predictions NetCDF file."""

    if writer.global_rank != 0:
        writer.write_or_wait(None, out_path)
        return out_path

    writer._clear_markers(out_path)
    dataset: xr.Dataset | None = None
    try:
        if out_path.exists():
            if config.allow_overwrite:
                out_path.unlink()
            else:
                raise SystemExit(
                    f"Refusing to overwrite existing prediction file: {out_path}. "
                    "Use a fresh output directory or pass --allow-overwrite-existing-files explicitly."
                )

        metadata = _metadata_from_ensemble(ensemble, config)
        dataset = build_prediction_dataset(
            x=ensemble.x_stack,
            y=ensemble.y_stack,
            y_pred=ensemble.y_pred_stack,
            x_interp=ensemble.x_interp_stack,
            lon_lres=ensemble.lon_lres,
            lat_lres=ensemble.lat_lres,
            lon_hres=ensemble.lon_hres,
            lat_hres=ensemble.lat_hres,
            weather_states=ensemble.weather_states,
            init_date=ensemble.init_date,
            lead_step_hours=ensemble.lead_step_hours,
            member_ids=ensemble.member_ids,
            include_member_views=not config.slim_output,
            metadata=metadata,
            source_bundle_paths=ensemble.source_bundle_paths,
        )
        errors = validate_predictions_dataset(dataset)
        if errors:
            raise ValueError("Invalid predictions dataset before write: " + "; ".join(errors))

        writer.write_or_wait(dataset, out_path)
    except BaseException as exc:
        writer._mark_failed(out_path, exc)
        raise
    finally:
        if dataset is not None:
            dataset.close()

    with xr.open_dataset(out_path) as written:
        errors = validate_predictions_dataset(written)
        if errors:
            raise ValueError("Invalid predictions dataset after write: " + "; ".join(errors))
    return out_path
