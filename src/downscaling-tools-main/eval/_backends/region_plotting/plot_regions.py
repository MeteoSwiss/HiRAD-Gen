from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from .local_plotting import LocalInferencePlotter, plot_x_y
from .plotting.config import (
    DEFAULT_MODEL_VARIABLES as CONFIG_DEFAULT_MODEL_VARIABLES,
    DEFAULT_WEATHER_STATES as CONFIG_DEFAULT_WEATHER_STATES,
    O96_INTERESTING_REGIONS as CONFIG_O96_INTERESTING_REGIONS,
    O1280_DETAIL_REGIONS as CONFIG_O1280_DETAIL_REGIONS,
    O1280_INTERESTING_REGIONS as CONFIG_O1280_INTERESTING_REGIONS,
    O2560_SHOWCASE_REGIONS as CONFIG_O2560_SHOWCASE_REGIONS,
    PREDICTION_REGION_BOXES as CONFIG_PREDICTION_REGION_BOXES,
    RENDER_DPI,
)
from .plotting.coordinate_utils import get_region_ds
from .plotting.datetime_utils import safe_datetime_str
from .plotting.manifest import write_manifest
from .plotting.metadata import sample_meta_title, step_meta_title
from .plotting.preprocessing import ensure_x_interp_for_plotting
from .plotting.variable_utils import supports_plot_variable
from manual_inference.data import DownscalingDatasetProcessor

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_VARIABLES = CONFIG_DEFAULT_MODEL_VARIABLES
DEFAULT_WEATHER_STATES = CONFIG_DEFAULT_WEATHER_STATES
O96_INTERESTING_REGIONS = CONFIG_O96_INTERESTING_REGIONS
O1280_INTERESTING_REGIONS = CONFIG_O1280_INTERESTING_REGIONS
O1280_DETAIL_REGIONS = CONFIG_O1280_DETAIL_REGIONS
O2560_SHOWCASE_REGIONS = CONFIG_O2560_SHOWCASE_REGIONS
PREDICTION_REGION_BOXES = CONFIG_PREDICTION_REGION_BOXES


def _safe_datetime_str(value) -> str:
    return safe_datetime_str(value)


def _sample_meta_title(ds_region: xr.Dataset, region_name: str, sample_idx: int) -> str:
    return sample_meta_title(ds_region, region_name, sample_idx)


def _step_meta_title(region_name: str, step, forecast_ref_time) -> str:
    return step_meta_title(region_name, step, forecast_ref_time)


def _select_prediction_variables(
    ds_pred: xr.Dataset,
    model_variables: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    requested_model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    selected_variables: list[str] = []
    for model_var in requested_model_variables:
        if not supports_plot_variable(ds_pred, model_var):
            continue
        selected_variables.append(model_var)
    if not selected_variables:
        raise ValueError(
            f"None of the requested model variables are available. Requested={requested_model_variables}"
        )

    available_weather_states = [str(v) for v in ds_pred["weather_state"].values.tolist()]
    selected_weather_states = [w for w in DEFAULT_WEATHER_STATES if w in available_weather_states]
    if not selected_weather_states:
        selected_weather_states = available_weather_states
    return selected_variables, selected_weather_states


def _write_suite_manifest(
    *,
    out_root: Path,
    suite_kind: str,
    predictions_path: Path,
    region_boxes: dict[str, list[float]],
    selected_variables: list[str],
    selected_weather_states: list[str],
    sample_index: int,
    ensemble_member_index: int,
    generated: list[str],
) -> Path:
    payload = {
        "suite_kind": suite_kind,
        "plot_style": "region_six_panel",
        "predictions_file": str(predictions_path),
        "out_dir": str(out_root),
        "regions": [
            {"name": name, "box": [float(v) for v in box]}
            for name, box in region_boxes.items()
        ],
        "model_variables": list(selected_variables),
        "weather_states": list(selected_weather_states),
        "sample_index": int(sample_index),
        "ensemble_member_index": int(ensemble_member_index),
        "generated_files": list(generated),
    }
    return write_manifest(out_root=out_root, payload=payload)


def _region_boxes_for_names(region_names: list[str] | None, *, grid: str) -> dict[str, list[float]]:
    if region_names:
        unknown = [name for name in region_names if name not in PREDICTION_REGION_BOXES]
        if unknown:
            raise ValueError(f"Unknown region names: {unknown}. Known regions: {sorted(PREDICTION_REGION_BOXES)}")
        return {name: PREDICTION_REGION_BOXES[name] for name in region_names}

    if grid == "O1280":
        return O1280_INTERESTING_REGIONS
    if grid == "O96":
        return O96_INTERESTING_REGIONS
    if grid == "O2560":
        return O2560_SHOWCASE_REGIONS

    raise ValueError(
        f"Cannot determine region suite: no region_names supplied and grid {grid!r} "
        "has no default suite. Pass explicit region_names or set the 'grid' attribute "
        "on the predictions dataset."
    )


def render_region_suite_from_predictions_file(
    *,
    predictions_nc: str | Path,
    out_dir: str | Path,
    region_names: list[str] | None = None,
    region_boxes: dict[str, list[float]] | None = None,
    model_variables: list[str] | None = None,
    weather_states: list[str] | None = None,
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    also_png: bool = True,
    suite_kind: str = "regions",
) -> list[str]:
    predictions_path = Path(predictions_nc).expanduser().resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    combined_pdf = out_root / "all_regions_plots.pdf"
    generated: list[str] = [str(combined_pdf)]

    suite_kind = str(suite_kind).strip().lower() or "regions"
    if suite_kind not in {"regions", "storm"}:
        raise ValueError(f"Unsupported suite_kind={suite_kind!r}; expected 'regions' or 'storm'.")

    requested_weather_states = weather_states or DEFAULT_WEATHER_STATES
    with xr.open_dataset(predictions_path) as ds_pred:
        if region_boxes is None:
            region_boxes = _region_boxes_for_names(
                region_names,
                grid=str(ds_pred.attrs.get("grid", "")).strip(),
            )

        if "sample" in ds_pred.dims:
            if not 0 <= sample_index < int(ds_pred.sizes["sample"]):
                raise IndexError(f"sample_index={sample_index} outside 0..{int(ds_pred.sizes['sample']) - 1}")
            ds_pred = ds_pred.isel(sample=sample_index)
        if "ensemble_member" in ds_pred.dims:
            if not 0 <= ensemble_member_index < int(ds_pred.sizes["ensemble_member"]):
                raise IndexError(
                    "ensemble_member_index="
                    f"{ensemble_member_index} outside 0..{int(ds_pred.sizes['ensemble_member']) - 1}"
                )
            ds_pred = ds_pred.isel(ensemble_member=ensemble_member_index)
        ds_pred = ensure_x_interp_for_plotting(ds_pred, predictions_path=predictions_path)

        selected_variables, selected_weather_states = _select_prediction_variables(
            ds_pred,
            model_variables=model_variables,
        )
        if weather_states:
            selected_weather_states = [w for w in requested_weather_states if w in selected_weather_states]
            if not selected_weather_states:
                selected_weather_states = requested_weather_states

        with PdfPages(combined_pdf) as pdf:
            for region_name, region_box in region_boxes.items():
                LOG.info("Plotting prediction region %s %s", region_name, region_box)
                region_ds = get_region_ds(ds_pred, region_box)
                title = _sample_meta_title(region_ds, region_name, sample_index)
                fig = plot_x_y(
                    ds_sample=region_ds,
                    list_model_variables=selected_variables,
                    weather_states=selected_weather_states,
                    title=title,
                )
                pdf.savefig(fig)
                plt.close(fig)

        manifest_path = _write_suite_manifest(
            out_root=out_root,
            suite_kind=suite_kind,
            predictions_path=predictions_path,
            region_boxes=region_boxes,
            selected_variables=selected_variables,
            selected_weather_states=selected_weather_states,
            sample_index=sample_index,
            ensemble_member_index=ensemble_member_index,
            generated=generated,
        )
        generated.append(str(manifest_path))

    return generated


def _save_custom_o1280_plots(
    *,
    ds_pred: xr.Dataset,
    out_pdf_path: Path,
    selected_variables: list[str],
    regions: dict[str, list[float]] = O1280_INTERESTING_REGIONS,
) -> None:
    weather_states = [w for w in DEFAULT_WEATHER_STATES if w in ds_pred["weather_state"].values]
    if out_pdf_path.exists():
        out_pdf_path.unlink()
    with PdfPages(out_pdf_path) as pdf:
        for region_name, region_box in regions.items():
            LOG.info("Plotting custom O1280 region %s %s", region_name, region_box)
            region_ds = get_region_ds(ds_pred, region_box)
            region_ds.attrs["region_name"] = region_name
            if "sample" in region_ds.dims:
                n_available = int(region_ds.sizes.get("sample", 0))
                n_to_plot = min(2, n_available)
                for sample_idx in range(n_to_plot):
                    title = _sample_meta_title(region_ds, region_name, sample_idx)
                    fig = plot_x_y(
                        ds_sample=region_ds.sel(sample=sample_idx),
                        list_model_variables=selected_variables,
                        weather_states=weather_states,
                        title=title,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
            else:
                sample_count = 0
                for step in region_ds.step.values:
                    for ft in np.atleast_1d(region_ds.forecast_reference_time.values):
                        if sample_count >= 2:
                            break
                        title = _step_meta_title(region_name, step, ft)
                        fig = plot_x_y(
                            ds_sample=region_ds.sel(step=step, forecast_reference_time=ft),
                            list_model_variables=selected_variables,
                            weather_states=weather_states,
                            title=title,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)
                        sample_count += 1
                    if sample_count >= 2:
                        break


def build_predictions_dataset_from_expver(
    *,
    expver: str,
    date: str,
    number: str,
    step: str,
    sfc_param: str,
    pl_param: str,
    level: str,
    low_res_reference_grib: str,
    high_res_reference_grib: str,
) -> xr.Dataset:
    ds_pred_processor = DownscalingDatasetProcessor(
        expver=expver,
        date=date,
        number=number,
        step=step,
        sfc_param=sfc_param,
        pl_param=pl_param,
        level=level,
        low_res_reference_grib=low_res_reference_grib,
        high_res_reference_grib=high_res_reference_grib,
    )
    LOG.info("Requesting MARS prediction dataset for expver=%s", expver)
    ds_pred_processor.request_predictions_dataset()
    LOG.info("Cleaning prediction dataset")
    ds_pred = ds_pred_processor.clean_predictions_dataset()
    LOG.info("Building target dataset")
    ds_target = ds_pred_processor.build_target_dataset()
    LOG.info("Building input dataset")
    ds_input = ds_pred_processor.build_input_dataset()
    LOG.info("Merging prediction/target/input datasets")
    return xr.merge([ds_pred, ds_target, ds_input])


def save_predictions_dataset(ds: xr.Dataset, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    ds.to_netcdf(out_path, mode="w")
    LOG.info("Saved predictions to %s", out_path)
    return out_path


def run_region_plots_from_predictions(
    *,
    run_parent_dir: str | Path,
    run_name: str,
    predictions_filename: str = "predictions.nc",
    model_variables: list[str] | None = None,
) -> None:
    run_parent_dir = Path(run_parent_dir)
    run_parent_dir.mkdir(parents=True, exist_ok=True)
    model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    predictions_path = run_parent_dir / run_name / predictions_filename
    with xr.open_dataset(predictions_path) as ds_pred:
        ds_pred = ensure_x_interp_for_plotting(ds_pred, predictions_path=predictions_path)
        selected_variables, selected_weather_states = _select_prediction_variables(
            ds_pred,
            model_variables=model_variables,
        )
        if ds_pred.attrs.get("grid") == "O1280":
            out_pdf = run_parent_dir / run_name / "all_regions_plots.pdf"
            LOG.info("Saving custom O1280 region plots for %s", run_name)
            _save_custom_o1280_plots(
                ds_pred=ds_pred,
                out_pdf_path=out_pdf,
                selected_variables=selected_variables,
            )
            return
    lip = LocalInferencePlotter(str(run_parent_dir), run_name, predictions_filename)
    LOG.info("Saving default region plots for %s", run_name)
    lip.save_plot(
        lip.regions,
        list_model_variables=selected_variables,
        weather_states=selected_weather_states,
    )


def save_local_plots(
    expver: str,
    date: str,
    number: str,
    step: str,
    sfc_param: str,
    pl_param: str,
    level: str,
    low_res_reference_grib: str,
    high_res_reference_grib: str,
    output_root: str = "/home/ecm5702/prepml",
) -> Path:
    run_dir = Path(output_root) / expver
    predictions_filename = "predictions.nc"
    ds = build_predictions_dataset_from_expver(
        expver=expver,
        date=date,
        number=number,
        step=step,
        sfc_param=sfc_param,
        pl_param=pl_param,
        level=level,
        low_res_reference_grib=low_res_reference_grib,
        high_res_reference_grib=high_res_reference_grib,
    )
    predictions_path = save_predictions_dataset(ds, run_dir / predictions_filename)
    run_region_plots_from_predictions(
        run_parent_dir=Path(output_root),
        run_name=expver,
        predictions_filename=predictions_filename,
    )
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictions from expver or render local region plots from predictions.")
    parser.add_argument("--expver", type=str, default="", help="Experiment version")
    parser.add_argument("--date", type=str, default="20230801/20230810", help="Dates")
    parser.add_argument("--number", type=str, default="1/2/3", help="Numbers")
    parser.add_argument("--step", type=str, default="48/120", help="Steps")
    parser.add_argument(
        "--sfc_param", type=str, default="2t/10u/10v/sp", help="Surface parameters"
    )
    parser.add_argument(
        "--pl_param", type=str, default="z/t/u/v", help="Pressure level parameters"
    )
    parser.add_argument(
        "--low_res_reference_grib",
        type=str,
        default="eefo_reference_o96-early-august.grib",
        help="Low resolution reference grib file",
    )
    parser.add_argument(
        "--high_res_reference_grib",
        type=str,
        default="enfo_reference_o320-early-august.grib",
        help="High resolution reference grib file",
    )
    parser.add_argument("--level", type=str, default="500/850", help="Levels")
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/ecm5702/prepml",
        help="Parent output directory where <expver>/predictions.nc and plots are stored.",
    )
    parser.add_argument(
        "--predictions-nc",
        type=str,
        default="",
        help="Existing predictions_*.nc file to render directly.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for direct predictions rendering mode.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="",
        help="Comma-separated region names for direct predictions rendering mode. Blank => default suite.",
    )
    parser.add_argument(
        "--region-boxes-json",
        type=str,
        default="",
        help='JSON object mapping region names to [lat_min, lat_max, lon_min, lon_max] boxes. '
             'When provided, bypasses PREDICTION_REGION_BOXES lookup entirely.',
    )
    parser.add_argument(
        "--model-variables",
        type=str,
        default=",".join(DEFAULT_MODEL_VARIABLES),
        help="Comma-separated model variables for direct predictions rendering mode.",
    )
    parser.add_argument(
        "--weather-states",
        type=str,
        default=",".join(DEFAULT_WEATHER_STATES),
        help="Comma-separated weather states for direct predictions rendering mode.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--suite-kind", default="regions", choices=["regions", "storm"])

    args = parser.parse_args()
    if args.predictions_nc:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --predictions-nc")
        explicit_boxes: dict[str, list[float]] | None = None
        if args.region_boxes_json:
            explicit_boxes = json.loads(args.region_boxes_json)
        explicit_names = [v.strip() for v in args.regions.split(",") if v.strip()] or None
        generated = render_region_suite_from_predictions_file(
            predictions_nc=args.predictions_nc,
            out_dir=args.out_dir,
            region_names=list(explicit_boxes) if explicit_boxes else explicit_names,
            region_boxes=explicit_boxes,
            model_variables=[v.strip() for v in args.model_variables.split(",") if v.strip()],
            weather_states=[v.strip() for v in args.weather_states.split(",") if v.strip()],
            sample_index=args.sample_index,
            ensemble_member_index=args.ensemble_member_index,
            also_png=args.also_png,
            suite_kind=args.suite_kind,
        )
        for path in generated:
            print(path)
        return

    if not args.expver:
        raise SystemExit("Either --predictions-nc or --expver is required.")
    save_local_plots(
        args.expver,
        args.date,
        args.number,
        args.step,
        args.sfc_param,
        args.pl_param,
        args.level,
        low_res_reference_grib=args.low_res_reference_grib,
        high_res_reference_grib=args.high_res_reference_grib,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
