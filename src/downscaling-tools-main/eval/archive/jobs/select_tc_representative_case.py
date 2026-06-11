#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import xarray as xr

from eval._backends.region_plotting.local_plotting import get_region_ds
from eval._backends.region_plotting.plot_regions import PREDICTION_REGION_BOXES
from eval._backends.tc.events import EVENTS

PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Select one representative TC case (date/step/member) from a predictions directory '
            'for storm-route comparison and intermediate plotting.'
        )
    )
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--bundle-dir', default='')
    parser.add_argument('--event', required=True)
    parser.add_argument('--metric', choices=['maxwind', 'minmsl'], default='maxwind')
    parser.add_argument('--region', default='')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--out-json', default='')
    return parser.parse_args()


def _prediction_files(predictions_dir: Path, event_name: str) -> list[tuple[Path, str, int]]:
    event = EVENTS[event_name]
    allowed_dates = {f"{event.year}{event.month}{day}" for day in event.dates}
    matches: list[tuple[Path, str, int]] = []
    for path in sorted(predictions_dir.glob('predictions_*.nc')):
        match = PRED_RE.match(path.name)
        if not match:
            continue
        date, step = match.group(1), int(match.group(2))
        if date in allowed_dates:
            matches.append((path, date, step))
    return matches


def _event_region(event_name: str, explicit_region: str) -> str:
    if explicit_region:
        return explicit_region
    center_region = f'{event_name}_center'
    if center_region in PREDICTION_REGION_BOXES:
        return center_region
    if event_name in PREDICTION_REGION_BOXES:
        return event_name
    raise KeyError(f'No plotting region registered for event={event_name!r}')


def _member_number(coord_value, member_index: int) -> int:
    try:
        raw = np.asarray(coord_value).item()
    except Exception:
        return member_index + 1
    if isinstance(raw, (np.integer, int)):
        val = int(raw)
        if 1 <= val <= 99:
            return val
        if 0 <= val <= 98:
            return val + 1
    text = str(raw)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    return member_index + 1


def _score_case(ds_region: xr.Dataset, metric: str) -> tuple[np.ndarray, np.ndarray]:
    msl = np.asarray(ds_region['y_pred'].sel(weather_state='msl').values, dtype=float)
    wind_u = np.asarray(ds_region['y_pred'].sel(weather_state='10u').values, dtype=float)
    wind_v = np.asarray(ds_region['y_pred'].sel(weather_state='10v').values, dtype=float)
    wind = np.sqrt(wind_u ** 2 + wind_v ** 2)

    if 'ensemble_member' not in ds_region.dims:
        msl = msl[None, :]
        wind = wind[None, :]

    min_msl = np.nanmin(msl, axis=-1)
    max_wind = np.nanmax(wind, axis=-1)
    if metric == 'maxwind':
        primary = max_wind
        secondary = -min_msl
    else:
        primary = -min_msl
        secondary = max_wind
    return primary, secondary


def select_case(*, predictions_dir: Path, bundle_dir: Path | None, event_name: str, metric: str, region_name: str, sample_index: int) -> dict:
    files = _prediction_files(predictions_dir, event_name)
    if not files:
        raise FileNotFoundError(f'No predictions_*.nc files matched event={event_name} under {predictions_dir}')

    region_box = PREDICTION_REGION_BOXES[region_name]
    best: dict | None = None

    for path, date, step in files:
        with xr.open_dataset(path) as ds:
            if 'sample' in ds.dims:
                if not 0 <= sample_index < int(ds.sizes['sample']):
                    raise IndexError(f'sample_index={sample_index} outside 0..{int(ds.sizes["sample"]) - 1}')
                ds = ds.isel(sample=sample_index)
            ds_region = get_region_ds(ds, region_box)
            if 'ensemble_member' in ds_region.dims:
                coord = ds_region['ensemble_member'].values
                n_members = int(ds_region.sizes['ensemble_member'])
            else:
                coord = np.asarray([1])
                n_members = 1
            primary, secondary = _score_case(ds_region, metric)
            for member_index in range(n_members):
                member_number = _member_number(coord[member_index] if n_members > 1 else coord[0], member_index)
                candidate = {
                    'event': event_name,
                    'metric': metric,
                    'region': region_name,
                    'region_box': [float(v) for v in region_box],
                    'predictions_file': str(path),
                    'predictions_file_name': path.name,
                    'date': date,
                    'step': int(step),
                    'sample_index': int(sample_index),
                    'ensemble_member_index': int(member_index),
                    'ensemble_member_number': int(member_number),
                    'min_msl_pa': float(np.nanmin(np.asarray(ds_region['y_pred'].sel(weather_state='msl').isel(ensemble_member=member_index).values if 'ensemble_member' in ds_region.dims else ds_region['y_pred'].sel(weather_state='msl').values, dtype=float))),
                    'max_wind10m_ms': float(np.nanmax(np.sqrt(
                        np.asarray(ds_region['y_pred'].sel(weather_state='10u').isel(ensemble_member=member_index).values if 'ensemble_member' in ds_region.dims else ds_region['y_pred'].sel(weather_state='10u').values, dtype=float) ** 2 +
                        np.asarray(ds_region['y_pred'].sel(weather_state='10v').isel(ensemble_member=member_index).values if 'ensemble_member' in ds_region.dims else ds_region['y_pred'].sel(weather_state='10v').values, dtype=float) ** 2
                    ))),
                    'selection_primary_score': float(primary[member_index]),
                    'selection_secondary_score': float(secondary[member_index]),
                }
                if best is None:
                    best = candidate
                    continue
                current_key = (candidate['selection_primary_score'], candidate['selection_secondary_score'])
                best_key = (best['selection_primary_score'], best['selection_secondary_score'])
                if current_key > best_key:
                    best = candidate

    assert best is not None
    best['bundle_dir'] = str(bundle_dir) if bundle_dir else ''
    if bundle_dir:
        bundle_path = bundle_dir / (
            f"eefo_o320_0001_date{best['date']}_time0000_mem{best['ensemble_member_number']:02d}_"
            f"step{best['step']:03d}h_input_bundle.nc"
        )
        best['bundle_file'] = str(bundle_path)
        best['bundle_exists'] = bundle_path.exists()
    else:
        best['bundle_file'] = ''
        best['bundle_exists'] = False
    best['case_tag'] = (
        f"{event_name}_date{best['date']}_step{best['step']:03d}_mem{best['ensemble_member_number']:02d}_{metric}"
    )
    return best


def main() -> None:
    args = _parse_args()
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    if not predictions_dir.is_dir():
        raise SystemExit(f'Predictions directory not found: {predictions_dir}')
    event_name = args.event.strip().lower()
    if event_name not in EVENTS:
        raise SystemExit(f'Unknown event {event_name!r}; known events: {sorted(EVENTS)}')
    region_name = _event_region(event_name, args.region.strip())
    bundle_dir = Path(args.bundle_dir).expanduser().resolve() if args.bundle_dir else None
    result = select_case(
        predictions_dir=predictions_dir,
        bundle_dir=bundle_dir,
        event_name=event_name,
        metric=args.metric,
        region_name=region_name,
        sample_index=int(args.sample_index),
    )
    text = json.dumps(result, indent=2) + '\n'
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding='utf-8')
        print(out_path)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
