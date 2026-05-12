#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval._backends.plot_intermediate.plot_intermediate import _predict_with_intermediates_single_member
from manual_inference.input_data_construction.bundle import (
    extract_target_from_bundle,
    load_inputs_from_bundle_numpy,
)
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.predict import (
    DEFAULT_EXTRA_ARGS_JSON,
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _resolve_ckpt_path,
    _resolve_device,
)


def _parse_steps(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _resolve_checkpoint(ckpt_ref: str, ckpt_root: str) -> tuple[Path, str]:
    candidate = Path(ckpt_ref).expanduser()
    if candidate.exists():
        return candidate.resolve(), str(candidate.resolve())
    resolved = _resolve_ckpt_path(ckpt_ref, ckpt_root)
    return Path(resolved).resolve(), ckpt_ref


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Generate intermediate-state NetCDF for one exact input bundle.')
    ap.add_argument('--bundle-nc', required=True)
    ap.add_argument('--out-nc', required=True)
    ap.add_argument('--ckpt-ref', required=True)
    ap.add_argument('--ckpt-root', default='/home/ecm5702/scratch/aifs/checkpoint')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num-gpus-per-model', type=int, default=0)
    ap.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    ap.add_argument('--validation-frequency', default='50h')
    ap.add_argument('--extra-args-json', default=DEFAULT_EXTRA_ARGS_JSON)
    ap.add_argument('--capture-steps', default='0,5,10,15,20,25,30,35,39')
    ap.add_argument('--include-init-state', action='store_true')
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    bundle_nc = Path(args.bundle_nc).expanduser().resolve()
    if not bundle_nc.exists():
        raise SystemExit(f'Bundle file not found: {bundle_nc}')
    out_nc = Path(args.out_nc).expanduser().resolve()
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    global_rank, local_rank, world_size = _get_parallel_info()
    num_gpus_per_model = int(args.num_gpus_per_model) if int(args.num_gpus_per_model) > 0 else int(world_size)
    requested_device = args.device
    if requested_device == 'cuda' and not torch.cuda.is_available():
        requested_device = 'cpu'
    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith('cuda'):
        torch.cuda.set_device(int(str(device).split(':')[1]))

    if num_gpus_per_model > 1 and world_size != num_gpus_per_model:
        raise SystemExit(
            f'Expected world_size={num_gpus_per_model} for model-parallel inference, got {world_size}.'
        )

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)
    ckpt_path, ckpt_label = _resolve_checkpoint(args.ckpt_ref, args.ckpt_root)

    interface, datamodule, _, _ = _load_objects(
        ckpt_path=ckpt_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
        num_gpus_per_model_override=num_gpus_per_model,
    )

    weather_states = list(datamodule.data_indices.model.output.name_to_index.keys())
    x_lres_np, x_hres_np, lon_lres, lat_lres, lon_hres, lat_hres = load_inputs_from_bundle_numpy(
        bundle_nc,
        datamodule.data_indices.data.input[0].name_to_index,
        datamodule.data_indices.data.input[1].name_to_index,
    )

    x_lres = torch.from_numpy(x_lres_np).to(device)[None, None, None, ...]
    x_hres = torch.from_numpy(x_hres_np).to(device)[None, None, None, ...]
    extra_args = json.loads(args.extra_args_json) if args.extra_args_json else {}

    with torch.inference_mode():
        final_pred, inter_steps, sampling_step_ids, x_interp_state = _predict_with_intermediates_single_member(
            interface=interface,
            x_in_lres=x_lres,
            x_in_hres=x_hres,
            extra_args=extra_args,
            model_comm_group=model_comm_group,
            capture_steps=_parse_steps(args.capture_steps),
            include_init_state=bool(args.include_init_state),
        )

    target_np, found_target_channels = extract_target_from_bundle(bundle_nc, weather_states)
    y_stack = None if target_np is None else target_np[None, None, :, :]
    x_stack = x_lres_np[None, None, :, :]
    y_pred_stack = final_pred[None, None, :, :]

    ds = build_predictions_dataset(
        x=x_stack,
        y=y_stack,
        y_pred=y_pred_stack,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        weather_states=weather_states,
        dates=None,
        member_ids=[0],
    )
    ds['inter_state'] = (
        ['sample', 'ensemble_member', 'sampling_step', 'grid_point_hres', 'weather_state'],
        inter_steps[None, None, ...],
    )
    ds['x_interp'] = (
        ['sample', 'ensemble_member', 'grid_point_hres', 'weather_state'],
        x_interp_state[None, None, ...],
    )
    ds = ds.assign_coords(sampling_step=sampling_step_ids)
    ds.attrs['intermediate_source'] = 'bundle'
    ds.attrs['bundle_nc'] = str(bundle_nc)
    ds.attrs['checkpoint_path'] = str(ckpt_path)
    ds.attrs['checkpoint_label'] = ckpt_label
    ds.attrs['sampling_config_json'] = args.extra_args_json
    ds.attrs['target_channels_found'] = int(found_target_channels)

    if global_rank == 0:
        ds.to_netcdf(out_nc)
        print(out_nc)


if __name__ == '__main__':
    main()
