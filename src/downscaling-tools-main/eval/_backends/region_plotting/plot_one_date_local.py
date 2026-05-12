from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from .local_plotting import plot_x_y
from .plotting.config import DEFAULT_MODEL_VARIABLES, DEFAULT_WEATHER_STATES, RENDER_DPI
from .plotting.coordinate_utils import default_region_for_grid, get_region_ds, infer_grid_type
from .plotting.manifest import write_manifest
from .plotting.metadata import build_run_plot_title
from .plotting.preprocessing import ensure_x_interp_for_plotting
from .plotting.variable_utils import supports_plot_variable

DEFAULT_LOCAL_PLOT_OUT_SUBDIR = "local_plots_one_date"


def _absolute_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _default_region_for_grid(grid: str) -> str:
    return default_region_for_grid(grid)


def _infer_grid(ds: xr.Dataset) -> str:
    """Infer grid from dataset attrs or hres dimension size."""
    return infer_grid_type(ds)


def _write_one_date_manifest(
    *,
    out_root: Path,
    run_root: Path,
    date: str,
    region: str,
    sample_index: int,
    ensemble_member_index: int,
    model_variables: list[str],
    weather_states: list[str],
    results: list[tuple[Path, Path | None]],
) -> Path:
    payload = {
        "suite_kind": "one_date_smoke",
        "plot_style": "region_six_panel",
        "run_root": str(run_root),
        "date": str(date),
        "region": str(region),
        "sample_index": int(sample_index),
        "ensemble_member_index": int(ensemble_member_index),
        "model_variables": list(model_variables),
        "weather_states": list(weather_states),
        "generated_files": [
            str(path)
            for pair in results
            for path in pair
            if path is not None
        ],
    }
    return write_manifest(out_root=out_root, payload=payload)


def _member_label(ds_member: xr.Dataset, ensemble_member_index: int) -> str:
    if "ensemble_member" not in ds_member.coords:
        return f"member{ensemble_member_index + 1:02d}"
    value = ds_member["ensemble_member"].values
    if getattr(value, "shape", ()) != ():
        flat = value.reshape(-1)
        value = flat[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return f"member{int(value):02d}"
    except Exception:
        return f"member{str(value)}"


def _select_member_dataset(ds: xr.Dataset, *, sample_index: int, ensemble_member_index: int) -> xr.Dataset:
    ds_member = ds
    if "sample" in ds_member.dims:
        if not 0 <= sample_index < int(ds_member.sizes["sample"]):
            raise IndexError(f"sample_index={sample_index} outside available range 0..{int(ds_member.sizes['sample']) - 1}")
        ds_member = ds_member.isel(sample=sample_index)
    if "ensemble_member" in ds_member.dims:
        if not 0 <= ensemble_member_index < int(ds_member.sizes["ensemble_member"]):
            raise IndexError(
                "ensemble_member_index="
                f"{ensemble_member_index} outside available range 0..{int(ds_member.sizes['ensemble_member']) - 1}"
            )
        ds_member = ds_member.isel(ensemble_member=ensemble_member_index)
    return ds_member


def _build_title(
    ds_member: xr.Dataset,
    *,
    region: str,
    run_label: str,
    sample_index: int,
    ensemble_member_index: int,
) -> str:
    return build_run_plot_title(
        ds_member,
        region=region,
        run_label=run_label,
        sample_index=sample_index,
        member_label=_member_label(ds_member, ensemble_member_index),
        ensemble_member_index=ensemble_member_index,
    )


def render_prediction_file(
    *,
    predictions_nc: str | Path,
    out: str | Path,
    also_png: str | Path = "",
    region: str = "auto",
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    run_label: str = "",
    model_variables: list[str] | None = None,
    weather_states: list[str] | None = None,
) -> tuple[Path, Path | None]:
    predictions_path = _absolute_path(predictions_nc)
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    requested_model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    requested_weather_states = weather_states or DEFAULT_WEATHER_STATES

    with xr.open_dataset(predictions_path) as ds:
        resolved_region = region
        if resolved_region == "auto":
            resolved_region = _default_region_for_grid(_infer_grid(ds))
        ds_member = _select_member_dataset(
            ds,
            sample_index=sample_index,
            ensemble_member_index=ensemble_member_index,
        )
        ds_member = ensure_x_interp_for_plotting(ds_member, predictions_path=predictions_path)

        selected_model_variables = [v for v in requested_model_variables if supports_plot_variable(ds_member, v)]
        if not selected_model_variables:
            raise ValueError(
                f"No requested model variables available in {predictions_path}. Requested={requested_model_variables}"
            )

        available_weather_states = [str(v) for v in ds_member["weather_state"].values.tolist()]
        selected_weather_states = [w for w in requested_weather_states if w in available_weather_states]
        if not selected_weather_states:
            selected_weather_states = available_weather_states
        if not selected_weather_states:
            raise ValueError(f"No weather states available in {predictions_path}")

        ds_region = get_region_ds(ds_member, resolved_region)
        title = _build_title(
            ds_member,
            region=resolved_region,
            run_label=run_label,
            sample_index=sample_index,
            ensemble_member_index=ensemble_member_index,
        )
        fig = plot_x_y(
            ds_sample=ds_region,
            list_model_variables=selected_model_variables,
            weather_states=selected_weather_states,
            title=title,
        )

    out_path = _absolute_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".pdf":
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig)
    else:
        fig.savefig(out_path, dpi=RENDER_DPI)

    png_path: Path | None = None
    if also_png:
        png_path = _absolute_path(also_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=RENDER_DPI)

    plt.close(fig)
    return out_path, png_path


def render_one_date_local_plots(
    *,
    run_root: str | Path,
    date: str,
    out_subdir: str = DEFAULT_LOCAL_PLOT_OUT_SUBDIR,
    region: str = "auto",
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    expected_count: int = 5,
    run_label: str = "",
    model_variables: list[str] | None = None,
    weather_states: list[str] | None = None,
) -> list[tuple[Path, Path | None]]:
    run_root_path = _absolute_path(run_root)
    predictions_dir = run_root_path / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions directory not found: {predictions_dir}")

    matches = sorted(predictions_dir.glob(f"predictions_{date}_step*.nc"))
    if expected_count > 0 and len(matches) != expected_count:
        raise ValueError(f"Expected {expected_count} prediction files for {date} but found {len(matches)} in {predictions_dir}")
    if not matches:
        raise FileNotFoundError(f"No prediction files matched predictions_{date}_step*.nc in {predictions_dir}")

    results: list[tuple[Path, Path | None]] = []
    effective_run_label = run_label or run_root_path.name
    effective_model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    effective_weather_states = weather_states or DEFAULT_WEATHER_STATES
    for predictions_path in matches:
        with xr.open_dataset(predictions_path) as ds:
            resolved_region = region
            if resolved_region == "auto":
                resolved_region = _default_region_for_grid(_infer_grid(ds))
            ds_member = _select_member_dataset(
                ds,
                sample_index=sample_index,
                ensemble_member_index=ensemble_member_index,
            )
            member = _member_label(ds_member, ensemble_member_index)
        step_dir = run_root_path / out_subdir / predictions_path.stem
        stem = f"{resolved_region}_{member}_baseline"
        out_pdf = step_dir / f"{stem}.pdf"
        out_png = step_dir / f"{stem}.png"
        results.append(
            render_prediction_file(
                predictions_nc=predictions_path,
                out=out_pdf,
                also_png=out_png,
                region=resolved_region,
                sample_index=sample_index,
                ensemble_member_index=ensemble_member_index,
                run_label=effective_run_label,
                model_variables=model_variables,
                weather_states=weather_states,
            )
        )
    out_root = run_root_path / out_subdir
    manifest_path = _write_one_date_manifest(
        out_root=out_root,
        run_root=run_root_path,
        date=date,
        region=region,
        sample_index=sample_index,
        ensemble_member_index=ensemble_member_index,
        model_variables=effective_model_variables,
        weather_states=effective_weather_states,
        results=results,
    )
    print(manifest_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render canonical non-TC local plots from one prediction file or from one evaluation date."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions-nc", default="", help="Single predictions file to plot.")
    mode.add_argument("--run-root", default="", help="Run root containing predictions/predictions_*.nc.")
    parser.add_argument("--date", default="", help="Required with --run-root; example: 20230826")
    parser.add_argument("--out", default="", help="Required with --predictions-nc.")
    parser.add_argument("--also-png", default="", help="Optional PNG output for --predictions-nc mode.")
    parser.add_argument("--out-subdir", default=DEFAULT_LOCAL_PLOT_OUT_SUBDIR)
    parser.add_argument("--region", default="auto")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--expected-count", type=int, default=5)
    parser.add_argument(
        "--model-variables",
        default=",".join(DEFAULT_MODEL_VARIABLES),
        help="Comma-separated model variables; defaults match plot_regions.py.",
    )
    parser.add_argument(
        "--weather-states",
        default=",".join(DEFAULT_WEATHER_STATES),
        help="Comma-separated weather states; defaults match plot_regions.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_variables = [v.strip() for v in args.model_variables.split(",") if v.strip()]
    weather_states = [w.strip() for w in args.weather_states.split(",") if w.strip()]

    if args.predictions_nc:
        if not args.out:
            raise SystemExit("--out is required with --predictions-nc")
        out_pdf, out_png = render_prediction_file(
            predictions_nc=args.predictions_nc,
            out=args.out,
            also_png=args.also_png,
            region=args.region,
            sample_index=args.sample_index,
            ensemble_member_index=args.ensemble_member_index,
            run_label=args.run_label,
            model_variables=model_variables,
            weather_states=weather_states,
        )
        print(out_pdf)
        if out_png:
            print(out_png)
        return

    if not args.date:
        raise SystemExit("--date is required with --run-root")
    results = render_one_date_local_plots(
        run_root=args.run_root,
        date=args.date,
        out_subdir=args.out_subdir,
        region=args.region,
        sample_index=args.sample_index,
        ensemble_member_index=args.ensemble_member_index,
        expected_count=args.expected_count,
        run_label=args.run_label,
        model_variables=model_variables,
        weather_states=weather_states,
    )
    for out_pdf, out_png in results:
        print(out_pdf)
        if out_png:
            print(out_png)


if __name__ == "__main__":
    main()
