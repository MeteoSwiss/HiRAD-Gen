from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from manual_inference.checkpoints import (
    adapt_config_hpc,
    get_checkpoint,
    get_datamodule,
    instantiate_config,
    to_omegaconf,
)
from manual_inference.config import BUNDLE_IMPLICIT_HRES_FEATURES
from manual_inference.config import DATASET_PATH_REWRITE_PREFIXES
from manual_inference.config import DEFAULT_CKPT_ROOT
from manual_inference.config import DEFAULT_EXPERIMENTS_DIR
from manual_inference.config import DEFAULT_EXTRA_ARGS_JSON

# Re-export for backward compatibility — external callers import these from here.
__all__ = ["DEFAULT_EXTRA_ARGS_JSON"]
from manual_inference.input_data_construction.bundle import extract_target_from_bundle
from manual_inference.input_data_construction.bundle import find_missing_explicit_hres_inputs
from manual_inference.input_data_construction.bundle import load_inputs_from_bundle_numpy
from manual_inference.input_data_construction.bundle import open_bundle_dataset
from manual_inference.input_data_construction.bundle import parse_channel_subset_csv as _parse_channel_subset_csv
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.dataset import OUTPUT_WEATHER_STATE_MODE_CHOICES
from manual_inference.prediction.dataset import resolve_output_weather_states
from manual_inference.prediction.utils import extract_filtered_input_from_output

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Matches any Jupiter .runtime_datasets/<run_id>/ prefix (e.g. o1280_406905).
# These are ephemeral symlink farms whose zarr filenames mirror the AG local mirror.
_JUPITER_RUNTIME_RE = re.compile(
    r"^/e/home/jusers/dumontlebrazidec1/jupiter/dev/\.runtime_datasets/[^/]+/"
)
_JUPITER_RUNTIME_LOCAL = "/home/mlx/ai-ml/datasets/"


def _rewrite_dataset_paths_in_place(node):
    if OmegaConf.is_config(node):
        container = OmegaConf.to_container(node, resolve=False)
        rewritten = _rewrite_dataset_paths_in_place(container)
        return OmegaConf.create(rewritten)
    if isinstance(node, dict):
        for k, v in list(node.items()):
            node[k] = _rewrite_dataset_paths_in_place(v)
        return node
    if isinstance(node, list):
        return [_rewrite_dataset_paths_in_place(v) for v in node]
    if isinstance(node, tuple):
        return tuple(_rewrite_dataset_paths_in_place(v) for v in node)
    if isinstance(node, str):
        # Imported checkpoints may carry absolute dataset roots from multiple remote
        # sites. Rewrite those to the canonical local mirror when the target exists.
        for remote_prefix, local_prefix in DATASET_PATH_REWRITE_PREFIXES:
            if node.startswith(remote_prefix):
                candidate = node.replace(remote_prefix, local_prefix, 1)
                if os.path.exists(candidate):
                    return candidate
        # Fallback: any Jupiter .runtime_datasets/<run_id>/ → local AG mirror.
        # Prevents breakage when new Jupiter training runs use a different run ID.
        m = _JUPITER_RUNTIME_RE.match(node)
        if m:
            candidate = _JUPITER_RUNTIME_LOCAL + node[m.end():]
            if os.path.exists(candidate):
                return candidate
        return node
    return node


def _split_ckpt_path(path: str) -> tuple[str, str, str, str]:
    ckpt_path = os.path.abspath(os.path.expanduser(path))
    name_ckpt = os.path.basename(ckpt_path)
    name_exp = os.path.basename(os.path.dirname(ckpt_path))
    dir_exp = os.path.dirname(os.path.dirname(ckpt_path))
    return ckpt_path, dir_exp, name_exp, name_ckpt


def _get_parallel_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return global_rank, local_rank, world_size


def _resolve_device(requested_device: str, local_rank: int) -> str:
    if requested_device == "cuda" and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return requested_device


def _init_model_comm_group(device: str, global_rank: int, world_size: int):
    if world_size <= 1:
        return None
    backend = "nccl" if str(device).startswith("cuda") else "gloo"
    if dist.is_initialized():
        return dist.new_group(list(range(world_size)))
    if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=global_rank,
        )
        return dist.new_group(list(range(world_size)))

    # Slurm fallback, mirrors distributed/utils.py behavior.
    from distributed.utils import init_parallel

    return init_parallel(device, global_rank, world_size)


def _resolve_ckpt_path(
    name_ckpt: str,
    ckpt_root: str,
    *,
    allow_inference_companion: bool = False,
) -> str:
    raw = os.path.expanduser(name_ckpt)
    raw_name = os.path.basename(raw)
    if raw_name.startswith("inference-") and raw_name.endswith(".ckpt"):
        if allow_inference_companion:
            return raw if os.path.isabs(raw) else os.path.join(os.path.expanduser(ckpt_root), raw)
        raise ValueError(
            "Pass the base checkpoint path, not the inference companion. "
            f"Got {raw_name}; expected the matching non-inference .ckpt file."
        )
    if os.path.isabs(raw):
        return raw
    root = os.path.expanduser(ckpt_root)
    if raw.endswith(".ckpt"):
        return os.path.join(root, raw)
    run_dir = os.path.join(root, raw)
    last_ckpt = os.path.join(run_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    ckpt_candidates = sorted(Path(run_dir).glob("*.ckpt"))
    primary_candidates = [p for p in ckpt_candidates if not p.name.startswith("inference-")]
    if len(primary_candidates) == 1:
        return str(primary_candidates[0])
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"No checkpoint file found under {run_dir}. Expected last.ckpt or one explicit *.ckpt file."
        )
    if not primary_candidates:
        names = ", ".join(p.name for p in ckpt_candidates[:5])
        raise FileNotFoundError(
            f"Only inference companion checkpoint(s) found under {run_dir} ({names}). "
            "Pass the matching base .ckpt path explicitly or restore the base checkpoint file."
        )
    names = ", ".join(p.name for p in primary_candidates[:5])
    raise FileNotFoundError(
        f"Multiple base checkpoint files found under {run_dir} ({names}). Pass an explicit --name-ckpt path."
    )




def load_objects(
    *,
    ckpt_path: str,
    device: str,
    validation_frequency: str,
    precision: str,
    num_gpus_per_model_override: int | None = None,
):
    ckpt_path, dir_exp, name_exp, name_ckpt = _split_ckpt_path(ckpt_path)
    checkpoint, config_checkpoint = get_checkpoint(dir_exp, name_exp, name_ckpt)
    # Use checkpoint-native data/dataloader config, only overriding local paths.
    local_paths_config = instantiate_config()
    config_checkpoint = adapt_config_hpc(config_checkpoint, local_paths_config)
    config_for_datamodule = to_omegaconf(config_checkpoint)
    config_for_datamodule = _rewrite_dataset_paths_in_place(config_for_datamodule)
    config_for_datamodule.dataloader.validation.frequency = validation_frequency
    if hasattr(config_for_datamodule.dataloader.validation, "num_workers"):
        config_for_datamodule.dataloader.validation.num_workers = 0
    # Bundle-based inference needs template tensors on the full grid. Some checkpoints
    # carry a multi-GPU-per-model training setting that shards template grids.
    if num_gpus_per_model_override is not None and hasattr(config_for_datamodule, "hardware"):
        config_for_datamodule.hardware.num_gpus_per_model = int(num_gpus_per_model_override)
    if num_gpus_per_model_override is not None and hasattr(config_for_datamodule.dataloader, "read_group_size"):
        config_for_datamodule.dataloader.read_group_size = int(num_gpus_per_model_override)

    inference_model = torch.load(
        os.path.join(dir_exp, name_exp, "inference-" + name_ckpt),
        map_location=torch.device(device),
        weights_only=False,
    ).to(device)
    if device == "cuda":
        if precision == "fp16":
            inference_model = inference_model.half()
        elif precision == "bf16":
            inference_model = inference_model.bfloat16()
    graph_data = inference_model.graph_data
    datamodule = get_datamodule(config_for_datamodule, graph_data)
    return inference_model, datamodule, dir_exp, name_exp


# Backward-compatible alias.
_load_objects = load_objects


def _parse_members(value: str, max_members: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(max_members))
    members = [int(v.strip()) for v in value.split(",") if v.strip()]
    invalid = [m for m in members if m < 0 or m >= max_members]
    if invalid:
        raise ValueError(
            f"Requested member(s) {invalid} out of range for available members [0, {max_members - 1}]."
        )
    return members


def _parse_output_weather_states(value: str) -> list[str] | None:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    return requested or None


def _validate_bundle_hres_contract(
    *,
    bundle=None,
    bundle_nc=None,
    name_to_idx_hres: dict[str, int],
    name_to_idx_out: dict[str, int],
) -> None:
    source = bundle if bundle is not None else bundle_nc
    if source is None:
        raise TypeError("Either bundle or bundle_nc must be provided")
    missing_hres = find_missing_explicit_hres_inputs(source, list(name_to_idx_hres.keys()))
    if not missing_hres:
        return

    missing_preview = ", ".join(missing_hres[:10])
    if len(missing_hres) > 10:
        missing_preview += f", ... ({len(missing_hres)} total)"
    overlap = sorted(set(name_to_idx_hres) & set(name_to_idx_out))
    supported_preview = ", ".join(sorted(BUNDLE_IMPLICIT_HRES_FEATURES))
    if set(name_to_idx_hres) == set(name_to_idx_out):
        overlap_note = (
            " The checkpoint's full HRES input channel set matches the model outputs exactly, "
            "which indicates target-like HRES inputs and is incompatible with forcings-only bundle inference."
        )
    elif overlap:
        overlap_note = (
            f" The checkpoint's HRES inputs overlap with model outputs on {len(overlap)} channel(s) "
            f"(for example: {', '.join(overlap[:5])})."
        )
    else:
        overlap_note = ""
    raise ValueError(
        "Bundle inference is incompatible with this checkpoint/bundle combination. "
        f"The bundle loader can synthesize only forcing/static HRES inputs ({supported_preview}); "
        f"the checkpoint also requires explicit HRES channel(s) that are missing from the bundle: {missing_preview}."
        f"{overlap_note} Rebuild the bundle with matching in_hres_* fields or use a checkpoint trained with forcings-only HRES inputs."
    )


def _predict_from_dataloader(
    *,
    inference_model,
    datamodule,
    device: str,
    idx: int,
    n_samples: int,
    members: Sequence[int],
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "all",
    output_weather_states: Sequence[str] | None = None,
    split: str = "valid",
):
    data = (datamodule.ds_train if split == "train" else datamodule.ds_valid).data
    x_in = np.asarray(data[idx : idx + n_samples][0])  # [dates, vars, ens, grid]
    x_in_hres = np.asarray(data[idx : idx + n_samples][1])
    y = np.asarray(data[idx : idx + n_samples][2])

    x_in = np.transpose(x_in, (0, 2, 3, 1))  # [sample, ens, grid, vars]
    x_in_hres = np.transpose(x_in_hres, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))

    name_to_idx_in = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index

    # Downscaling models expect full lres input (including forcings) — the model's
    # normalizer has parameters for all input channels.  Only filter the *output
    # reference* arrays (x_out, y_out) later; keep x_in unfiltered for predict_step.
    x_in_full = x_in  # keep full-channel copy for the model

    lon_lres = np.asarray(data.longitudes[0])
    lat_lres = np.asarray(data.latitudes[0])
    lon_hres = np.asarray(data.longitudes[2])
    lat_hres = np.asarray(data.latitudes[2])
    dates = np.asarray(data.dates[idx : idx + n_samples])
    full_weather_states = list(name_to_idx_out.keys())
    weather_states, selected_indices = resolve_output_weather_states(
        weather_states=full_weather_states,
        mode=output_weather_state_mode,
        explicit_weather_states=output_weather_states,
    )

    y_pred = np.zeros(
        (n_samples, len(members), lon_hres.shape[0], len(weather_states)),
        dtype=np.float32,
    )

    amp_enabled = device == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    for i_sample in range(n_samples):
        for j, m in enumerate(members):
            x_l = torch.from_numpy(x_in_full[i_sample, m]).to(device)
            x_h = torch.from_numpy(x_in_hres[i_sample, m]).to(device)
            x_l = x_l[None, None, None, ...]
            x_h = x_h[None, None, None, ...]
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    pred = inference_model.predict_step(
                        x_l,
                        x_h,
                        model_comm_group=model_comm_group,
                        extra_args=extra_args,
                    )
            y_pred[i_sample, j] = (
                pred[0, 0, 0][..., selected_indices].detach().cpu().numpy().astype(np.float32)
            )

    if not members:
        raise ValueError("No members selected. Pass at least one member id.")
    # For output comparison, filter x_in to output-matching variables
    x_in_filtered, _ = extract_filtered_input_from_output(
        x_in, name_to_idx_in, name_to_idx_out
    )
    x_out = x_in_filtered[:, members, :, :][..., selected_indices]
    y_out = y[:, members, :, :][..., selected_indices]
    return (
        x_out,
        y_out,
        y_pred,
        lon_lres,
        lat_lres,
        lon_hres,
        lat_hres,
        weather_states,
        dates,
    )


def predict_from_bundle(
    *,
    inference_model,
    datamodule,
    device: str,
    bundle_nc: str,
    member_index: int,
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "all",
    output_weather_states: Sequence[str] | None = None,
):
    if member_index != 0:
        raise ValueError(
            "Bundle inputs are single-member bundles. Use member_index=0 and select the desired "
            "ensemble member while building the bundle."
        )

    name_to_idx_lres = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_hres = datamodule.data_indices.data.input[1].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index

    bundle = open_bundle_dataset(bundle_nc)
    try:
        _validate_bundle_hres_contract(
            bundle=bundle,
            name_to_idx_hres=name_to_idx_hres,
            name_to_idx_out=name_to_idx_out,
        )
        (
            x_lres_np,
            x_hres_np,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
        ) = load_inputs_from_bundle_numpy(
            bundle,
            name_to_idx_lres,
            name_to_idx_hres,
        )
        x_in = torch.from_numpy(x_lres_np).to(device)[None, None, None, ...]
        x_in_hres = torch.from_numpy(x_hres_np).to(device)[None, None, None, ...]

        amp_enabled = device == "cuda" and precision in {"fp16", "bf16"}
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                pred = inference_model.predict_step(
                    x_in[0:1],
                    x_in_hres[0:1],
                    model_comm_group=model_comm_group,
                    extra_args=extra_args,
                )

        weather_states_full = list(name_to_idx_out.keys())
        weather_states, selected_indices = resolve_output_weather_states(
            weather_states=weather_states_full,
            mode=output_weather_state_mode,
            explicit_weather_states=output_weather_states,
        )

        x_np = x_in[0, 0, 0].detach().cpu().numpy().astype(np.float32)
        pred_np = pred[0, 0, 0][..., selected_indices].detach().cpu().numpy().astype(np.float32)
        dates = None

        x_np, _ = extract_filtered_input_from_output(
            x_np, datamodule.data_indices.data.input[0].name_to_index, name_to_idx_out
        )
        x_np = x_np[..., selected_indices]

        y_np = None
        target_np, found_target_channels = extract_target_from_bundle(bundle, weather_states)
        if target_np is not None:
            y_np = target_np[None, None, ...]
            if found_target_channels < len(weather_states):
                print(
                    f"Bundle target coverage: {found_target_channels}/{len(weather_states)} weather states "
                    f"(missing channels will be NaN in y)."
                )

        return (
            x_np[None, ...],
            y_np,
            pred_np[None, None, ...],
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        )
    finally:
        try:
            bundle.close()
        except Exception:
            pass


# Backward-compatible alias.
_predict_from_bundle = predict_from_bundle


def _compute_x_interp_for_export(
    *,
    inference_model,
    x: np.ndarray,
    device: str,
    model_comm_group,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 3:
        x_arr = x_arr[:, None, ...]
    elif x_arr.ndim != 4:
        raise ValueError(f"Unsupported x shape for x_interp export: {x_arr.shape}")

    x_tensor = torch.from_numpy(x_arr).to(device)
    with torch.inference_mode():
        if hasattr(inference_model, "interpolate_down"):
            member_interp = []
            for member_idx in range(x_tensor.shape[1]):
                member_x = x_tensor[:, member_idx, ...]
                try:
                    interp = inference_model.interpolate_down(member_x, grad_checkpoint=False)
                except TypeError:
                    interp = inference_model.interpolate_down(member_x)
                if interp.ndim != 3:
                    raise ValueError(
                        f"Unexpected interpolate_down output shape {tuple(interp.shape)} for member {member_idx}"
                    )
                member_interp.append(interp)
            x_interp = torch.stack(member_interp, dim=1)
        else:
            model = getattr(inference_model, "model", inference_model)
            if not hasattr(model, "apply_interpolate_to_high_res"):
                raise RuntimeError(
                    "Inference model cannot export x_interp: missing interpolate_down and apply_interpolate_to_high_res."
                )
            try:
                x_interp = model.apply_interpolate_to_high_res(
                    x_tensor,
                    grid_shard_shapes=None,
                    model_comm_group=model_comm_group,
                )
            except TypeError:
                x_interp = model.apply_interpolate_to_high_res(x_tensor)

    return x_interp.detach().cpu().numpy().astype(np.float32)



def _parse_json(value: str) -> dict:
    if not value:
        return {}
    return json.loads(value)


def _fail_if_missing_truth(*, y: np.ndarray | None, context: str) -> None:
    if y is None:
        raise SystemExit(
            "Missing target truth `y` in predictions output. "
            f"Context={context}. Rebuild bundles with complete target_hres_* fields."
        )


def _coerce_missing_truth_to_nan(
    *,
    y: np.ndarray | None,
    y_pred: np.ndarray,
    context: str,
    allow_missing_target_unsafe: bool,
) -> tuple[np.ndarray, bool]:
    if y is not None:
        return y, False
    if not allow_missing_target_unsafe:
        _fail_if_missing_truth(y=y, context=context)
    print(
        "WARNING: missing target truth `y` in predictions output. "
        f"Context={context}. Writing all-NaN y because --allow-missing-target-unsafe was set. "
        "Treat this artifact as prediction-only and non-canonical for truth-aware evaluation."
    )
    return np.full_like(y_pred, np.nan, dtype=np.float32), True


def _validate_output_path(
    *,
    out_path: Path,
    allow_existing_output_dir: bool,
) -> None:
    resolved = out_path.expanduser().resolve()
    parent = resolved.parent

    # Guard against accidental nested run layouts like <old_run>/<new_run>/...
    parent_name = parent.name
    grandparent_name = parent.parent.name if parent.parent != parent else ""
    if _RUN_NAME_RE.fullmatch(parent_name) and _RUN_NAME_RE.fullmatch(grandparent_name):
        if parent.parent.exists() and (parent.parent / "logs").is_dir():
            raise SystemExit(
                f"Unsafe nested output path detected: {resolved}. "
                "Refusing to place a new run folder under an existing run folder."
            )

    if parent.exists():
        if not allow_existing_output_dir and any(parent.iterdir()):
            raise SystemExit(
                f"Output directory already exists and is not empty: {parent}. "
                "Use a fresh run folder or pass --allow-existing-output-dir explicitly."
            )
    else:
        parent.mkdir(parents=True, exist_ok=False)

    if resolved.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output file: {resolved}. "
            "Use a fresh run folder or rename the output file."
        )


def main() -> None:
    ckpt_root_default = os.environ.get(
        "AIFS_CKPT_ROOT", DEFAULT_CKPT_ROOT
    )
    parser = argparse.ArgumentParser(description="Generate predictions.nc from a checkpoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("from-dataloader", help="Use dataloader inputs.")
    p_dl.add_argument("--name-ckpt", required=True)
    p_dl.add_argument("--ckpt-root", default=ckpt_root_default)
    p_dl.add_argument("--device", default="cuda")
    p_dl.add_argument("--idx", type=int, default=0)
    p_dl.add_argument("--n-samples", type=int, default=1)
    p_dl.add_argument("--members", default="0")
    p_dl.add_argument("--validation-frequency", default="50h")
    p_dl.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_dl.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_dl.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="all",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    p_dl.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    p_dl.add_argument(
        "--slim-output",
        action="store_true",
        help="Write only canonical x/y/y_pred ensemble arrays and skip duplicate x_*/y_*/y_pred_* member views.",
    )
    p_dl.add_argument("--out", default="")
    p_dl.add_argument(
        "--split",
        choices=["valid", "train"],
        default="valid",
        help="Which data split to sample from (default: valid).",
    )
    p_dl.add_argument(
        "--debug-from-dataloader",
        action="store_true",
        help="Required safety switch: from-dataloader is debug-only.",
    )
    p_dl.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into a pre-existing non-empty output directory.",
    )

    p_bundle = sub.add_parser("from-bundle", help="Use a prebuilt input bundle NetCDF.")
    p_bundle.add_argument("--name-ckpt", required=True)
    p_bundle.add_argument("--ckpt-root", default=ckpt_root_default)
    p_bundle.add_argument("--device", default="cuda")
    p_bundle.add_argument("--bundle-nc", required=True)
    p_bundle.add_argument("--member-index", type=int, default=0)
    p_bundle.add_argument("--validation-frequency", default="50h")
    p_bundle.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_bundle.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_bundle.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="all",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    p_bundle.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    p_bundle.add_argument(
        "--slim-output",
        action="store_true",
        help="Write only canonical x/y/y_pred ensemble arrays and skip duplicate x_*/y_*/y_pred_* member views.",
    )
    p_bundle.add_argument("--out", default="")
    p_bundle.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into a pre-existing non-empty output directory.",
    )
    p_bundle.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow missing target_hres_* truth in bundle predictions by writing "
            "all-NaN y. Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )

    p_bundle_build = sub.add_parser("build-bundle", help="Create input bundle from GRIB.")
    p_bundle_build.add_argument("--lres-sfc-grib", required=True)
    p_bundle_build.add_argument(
        "--lres-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra low-resolution surface GRIB to merge before bundle creation.",
    )
    p_bundle_build.add_argument("--lres-pl-grib", required=True)
    p_bundle_build.add_argument("--hres-grib", required=True)
    p_bundle_build.add_argument(
        "--hres-static-grib",
        default="",
        help="Optional GRIB file used only for high-resolution static fields such as z and lsm.",
    )
    p_bundle_build.add_argument("--target-sfc-grib", default="")
    p_bundle_build.add_argument(
        "--target-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra target surface GRIB to merge before bundle creation.",
    )
    p_bundle_build.add_argument("--target-pl-grib", default="")
    p_bundle_build.add_argument(
        "--lres-sfc-channels",
        default="",
        help="Optional CSV override for low-resolution surface bundle channels.",
    )
    p_bundle_build.add_argument(
        "--lres-pl-channels",
        default="",
        help="Optional CSV override for low-resolution pressure-level bundle channels.",
    )
    p_bundle_build.add_argument(
        "--target-sfc-channels",
        default="",
        help="Optional CSV override for target high-resolution surface bundle channels.",
    )
    p_bundle_build.add_argument(
        "--target-pl-channels",
        default="",
        help="Optional CSV override for target high-resolution pressure-level bundle channels.",
    )
    p_bundle_build.add_argument("--allow-missing-target", action="store_true")
    p_bundle_build.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow creating bundle without target_hres_* fields. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    p_bundle_build.add_argument("--out", required=True)
    p_bundle_build.add_argument("--step-hours", type=int, default=None)
    p_bundle_build.add_argument("--member", type=int, default=None)

    args = parser.parse_args()
    global_rank, local_rank, world_size = _get_parallel_info()

    if args.cmd == "build-bundle":
        from manual_inference.input_data_construction.bundle import (
            build_input_bundle_from_grib,
        )
        if args.allow_missing_target:
            raise SystemExit(
                "--allow-missing-target is deprecated. "
                "Use --allow-missing-target-unsafe for an explicit prediction-only escape hatch."
            )

        out = build_input_bundle_from_grib(
            lres_sfc_grib=args.lres_sfc_grib,
            lres_sfc_extra_gribs=args.lres_sfc_extra_grib,
            lres_pl_grib=args.lres_pl_grib,
            hres_grib=args.hres_grib,
            hres_static_grib=args.hres_static_grib or None,
            out_nc=args.out,
            step_hours=args.step_hours,
            member=args.member,
            target_sfc_grib=args.target_sfc_grib or None,
            target_sfc_extra_gribs=args.target_sfc_extra_grib,
            target_pl_grib=args.target_pl_grib or None,
            require_target_fields=not args.allow_missing_target_unsafe,
            lres_sfc_channels=_parse_channel_subset_csv(args.lres_sfc_channels),
            lres_pl_channels=_parse_channel_subset_csv(args.lres_pl_channels),
            target_sfc_channels=_parse_channel_subset_csv(args.target_sfc_channels),
            target_pl_channels=_parse_channel_subset_csv(args.target_pl_channels),
        )
        print(f"Saved bundle: {out}")
        return

    resolved_ckpt = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but CUDA is not available on this host. "
            "Refusing to fall back to CPU for diffusion sampling."
        )
    args.device = _resolve_device(args.device, local_rank)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(int(str(args.device).split(":")[1]))
    model_comm_group = _init_model_comm_group(args.device, global_rank, world_size)

    inference_model, datamodule, dir_exp, name_exp = _load_objects(
        ckpt_path=resolved_ckpt,
        device=args.device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
    )

    extra_args = _parse_json(args.extra_args_json)
    output_weather_states = _parse_output_weather_states(getattr(args, "output_weather_states", ""))
    if args.cmd == "from-dataloader":
        if not args.debug_from_dataloader:
            raise SystemExit(
                "from-dataloader is debug-only in the new stack. "
                "Pass --debug-from-dataloader to run it intentionally."
            )
        data = (datamodule.ds_train if args.split == "train" else datamodule.ds_valid).data
        max_members = int(np.asarray(data[args.idx : args.idx + 1][0]).shape[2])
        members = _parse_members(args.members, max_members)
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_dataloader(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            idx=args.idx,
            n_samples=args.n_samples,
            members=members,
            extra_args=extra_args,
            precision=args.precision,
            model_comm_group=model_comm_group,
            output_weather_state_mode=args.output_weather_state_mode,
            output_weather_states=output_weather_states,
            split=args.split,
        )
        member_ids = members
        y, used_missing_target_unsafe = _coerce_missing_truth_to_nan(
            y=y,
            y_pred=y_pred,
            context="from-dataloader",
            allow_missing_target_unsafe=bool(getattr(args, "allow_missing_target_unsafe", False)),
        )
    elif args.cmd == "from-bundle":
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_bundle(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            bundle_nc=args.bundle_nc,
            member_index=args.member_index,
            extra_args=extra_args,
            precision=args.precision,
            model_comm_group=model_comm_group,
            output_weather_state_mode=args.output_weather_state_mode,
            output_weather_states=output_weather_states,
        )
        member_ids = [args.member_index]
        y, used_missing_target_unsafe = _coerce_missing_truth_to_nan(
            y=y,
            y_pred=y_pred,
            context="from-bundle",
            allow_missing_target_unsafe=bool(getattr(args, "allow_missing_target_unsafe", False)),
        )
    else:
        raise SystemExit("Unknown command")

    x_interp = _compute_x_interp_for_export(
        inference_model=inference_model,
        x=x,
        device=args.device,
        model_comm_group=model_comm_group,
    )

    ds = build_predictions_dataset(
        x=x,
        y=y,
        y_pred=y_pred,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        weather_states=weather_states,
        dates=dates,
        member_ids=member_ids,
        x_interp=x_interp,
        include_member_views=not getattr(args, "slim_output", False),
    )
    if used_missing_target_unsafe:
        ds.attrs["missing_target_policy"] = "all_nan_due_to_allow_missing_target_unsafe"
    ds.attrs["checkpoint_path"] = resolved_ckpt
    ds.attrs["sampling_config_json"] = args.extra_args_json
    ds.attrs["validation_frequency"] = args.validation_frequency
    ds.attrs["x_interp_exported"] = 1
    ds.attrs["output_weather_state_mode"] = args.output_weather_state_mode
    ds.attrs["output_weather_states"] = ",".join(weather_states)
    ds.attrs["slim_output"] = int(bool(getattr(args, "slim_output", False)))

    out_path = args.out
    if not out_path:
        experiments_dir = os.environ.get("AIFS_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)
        out_path = os.path.join(
            experiments_dir, name_exp, "predictions.nc"
        )
    out_path = Path(out_path)
    _validate_output_path(
        out_path=out_path,
        allow_existing_output_dir=bool(getattr(args, "allow_existing_output_dir", False)),
    )
    if global_rank == 0:
        ds.to_netcdf(out_path)
        print(f"Saved predictions: {out_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
