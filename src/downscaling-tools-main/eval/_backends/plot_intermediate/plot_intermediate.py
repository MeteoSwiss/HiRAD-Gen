from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import xarray as xr

from eval._backends.region_plotting.local_plotting import get_region_ds
from manual_inference.config import DEFAULT_EXTRA_ARGS_JSON
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.predict import (
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _parse_json,
    _resolve_ckpt_path,
    _resolve_device,
)
from manual_inference.prediction.utils import extract_filtered_input_from_output

try:
    from anemoi.models.distributed.graph import gather_tensor
    from anemoi.models.distributed.shapes import apply_shard_shapes
    from anemoi.models.samplers import diffusion_samplers
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import anemoi diffusion samplers. "
        "Ensure anemoi-core/anemoi-models is available in PYTHONPATH."
    ) from exc

try:
    from anemoi.training.diagnostics.maps import Coastlines
except Exception:  # pragma: no cover
    Coastlines = None

try:
    from cartopy import crs as ccrs
except Exception:  # pragma: no cover
    ccrs = None


NOISE_KEYS = {
    "schedule_type",
    "sigma_max",
    "sigma_min",
    "rho",
    "num_steps",
    "sigma_transition",
    "num_steps_high",
    "num_steps_low",
    "high_schedule_type",
    "low_schedule_type",
    "rho_high",
    "rho_low",
}
SAMPLER_KEYS = {"sampler", "S_churn", "S_min", "S_max", "S_noise"}


def select_sampling_steps(total_steps: int, max_panels: int) -> list[int]:
    if total_steps <= 0:
        return []
    n = min(total_steps, max_panels)
    return sorted({int(i) for i in np.linspace(0, total_steps - 1, n)})


def resolve_capture_steps(
    total_steps: int,
    explicit_steps: Sequence[int] | None = None,
    capture_max_steps: int = 0,
) -> list[int]:
    if total_steps <= 0:
        return []
    if explicit_steps:
        valid = sorted({int(s) for s in explicit_steps if 0 <= int(s) < total_steps})
        if not valid:
            raise ValueError("No valid capture steps within [0, total_steps).")
        if (total_steps - 1) not in valid:
            valid.append(total_steps - 1)
        return sorted(valid)
    if capture_max_steps and capture_max_steps > 0:
        return select_sampling_steps(total_steps, capture_max_steps)
    return list(range(total_steps))


def _parse_steps_csv(steps: str) -> list[int]:
    if not steps:
        return []
    return [int(x.strip()) for x in steps.split(",") if x.strip()]


def _split_sampling_args(extra_args: dict) -> tuple[dict, dict]:
    noise = {k: v for k, v in extra_args.items() if k in NOISE_KEYS}
    sampler = {k: v for k, v in extra_args.items() if k in SAMPLER_KEYS}
    return noise, sampler


def _build_sigmas(model, x_in_interp_to_hres: torch.Tensor, noise_scheduler_params: dict) -> torch.Tensor:
    noise_scheduler_config = dict(model.inference_defaults.noise_scheduler)
    noise_scheduler_config.update(noise_scheduler_params)
    schedule_type = noise_scheduler_config.pop("schedule_type")
    if schedule_type not in diffusion_samplers.NOISE_SCHEDULERS:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    scheduler = diffusion_samplers.NOISE_SCHEDULERS[schedule_type](**noise_scheduler_config)
    return scheduler.get_schedule(x_in_interp_to_hres.device, torch.float64)


def _sample_heun_with_intermediates(
    *,
    model,
    x_in_interp: torch.Tensor,
    x_in_hres: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    model_comm_group,
    grid_shard_shapes,
    sampler_params: dict,
    capture_steps: set[int] | None = None,
) -> list[torch.Tensor]:
    cfg = dict(model.inference_defaults.diffusion_sampler)
    cfg.update(sampler_params)
    cfg.pop("sampler", None)

    s_churn = cfg.get("S_churn", 0.0)
    s_min = cfg.get("S_min", 0.0)
    s_max = cfg.get("S_max", float("inf"))
    s_noise = cfg.get("S_noise", 1.0)
    dtype = sigmas.dtype
    eps_prec = cfg.get("eps_prec", 1e-10)

    batch_size = x_in_interp.shape[0]
    ensemble_size = x_in_interp.shape[2]
    num_steps = len(sigmas) - 1

    out_steps: list[torch.Tensor] = []
    for i in range(num_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        apply_churn = s_min <= sigma_i <= s_max and s_churn > 0.0
        if apply_churn:
            gamma = min(
                s_churn / num_steps,
                torch.sqrt(torch.tensor(2.0, dtype=sigma_i.dtype)) - 1,
            )
            sigma_effective = sigma_i + gamma * sigma_i
            epsilon = torch.randn_like(y) * s_noise
            y = y + torch.sqrt(sigma_effective**2 - sigma_i**2) * epsilon
        else:
            sigma_effective = sigma_i

        sigma_eff_exp = sigma_effective.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
        d1 = model.fwd_with_preconditioning(
            x_in_interp,
            x_in_hres,
            y.to(dtype=x_in_interp.dtype),
            sigma_eff_exp.to(x_in_interp.dtype),
            model_comm_group,
            grid_shard_shapes,
        ).to(dtype)

        d = (y - d1) / (sigma_effective + eps_prec)
        y_next = y + (sigma_next - sigma_effective) * d

        if sigma_next > eps_prec:
            sigma_next_exp = sigma_next.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
            d2 = model.fwd_with_preconditioning(
                x_in_interp,
                x_in_hres,
                y_next.to(dtype=x_in_interp.dtype),
                sigma_next_exp.to(dtype=x_in_interp.dtype),
                model_comm_group,
                grid_shard_shapes,
            ).to(dtype)
            d_prime = (y_next - d2) / (sigma_next + eps_prec)
            y = y + (sigma_next - sigma_effective) * (d + d_prime) / 2
        else:
            y = y_next

        if (capture_steps is None) or (i in capture_steps):
            out_steps.append(y.detach().to(device="cpu", dtype=torch.float32).clone())

    return out_steps


def _sample_dpmpp_2m_with_intermediates(
    *,
    model,
    x_in_interp: torch.Tensor,
    x_in_hres: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    model_comm_group,
    grid_shard_shapes,
    capture_steps: set[int] | None = None,
) -> list[torch.Tensor]:
    y = y.to(x_in_interp.dtype)
    sigmas = sigmas.to(x_in_interp.dtype)

    batch_size = x_in_interp.shape[0]
    ensemble_size = x_in_interp.shape[2]
    num_steps = len(sigmas) - 1
    old_denoised = None

    out_steps: list[torch.Tensor] = []
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_exp = sigma.view(1, 1, 1, 1).expand(batch_size, ensemble_size, 1, 1)
        denoised = model.fwd_with_preconditioning(
            x_in_interp,
            x_in_hres,
            y,
            sigma_exp,
            model_comm_group,
            grid_shard_shapes,
        )

        if sigma_next == 0:
            y = denoised
            if (capture_steps is None) or (i in capture_steps):
                out_steps.append(y.detach().to(device="cpu", dtype=torch.float32).clone())
            break

        t = -torch.log(sigma + 1e-10)
        t_next = -torch.log(sigma_next + 1e-10) if sigma_next > 0 else float("inf")
        h = t_next - t

        if old_denoised is None:
            x0 = denoised
            y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * x0
        else:
            h_last = -torch.log(sigmas[i - 1] + 1e-10) - t if i > 0 else h
            r = h_last / h
            x0 = denoised
            x0_last = old_denoised
            coeff1 = 1 + 1 / (2 * r)
            coeff2 = -1 / (2 * r)
            d = coeff1 * x0 + coeff2 * x0_last
            y = (sigma_next / sigma) * y - (torch.exp(-h) - 1) * d

        old_denoised = denoised
        if (capture_steps is None) or (i in capture_steps):
            out_steps.append(y.detach().to(device="cpu", dtype=torch.float32).clone())

    return out_steps


def _predict_with_intermediates_single_member(
    *,
    interface,
    x_in_lres: torch.Tensor,
    x_in_hres: torch.Tensor,
    extra_args: dict,
    model_comm_group,
    capture_steps: Sequence[int] | None = None,
    include_init_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = interface.model
    if not hasattr(model, "_before_sampling"):
        # OLD interface path (e.g. DownscalingModelInterface in anemoi_2024):
        # manually run default_sampler with return_intermediate enabled.
        nsteps = int(extra_args.get("num_steps", 40))
        sigma_min = float(extra_args.get("sigma_min", 0.03))
        sigma_max = float(extra_args.get("sigma_max", 80.0))
        rho = float(extra_args.get("rho", 7.0))
        capture_steps_resolved = resolve_capture_steps(
            total_steps=nsteps,
            explicit_steps=capture_steps,
            capture_max_steps=0,
        )

        x_in_4d = x_in_lres[:, 0, ...]
        x_hres_4d = x_in_hres[:, 0, ...]
        x_interp_to_hres = interface.interpolate_down(x_in_lres[:, 0, 0, ...], grad_checkpoint=False)[:, None, ...]
        x_interp_state = x_interp_to_hres[0, 0].detach().cpu().numpy().astype(np.float32)

        if getattr(interface, "model_version", "one_encoder") == "two_encoder":
            x_proc = interface.pre_processors_state(x_in_4d, "input_lres", in_place=False)
            x_proc = x_proc[..., interface.data_indices.data.input[0].full]
        else:
            x_proc = interface.pre_processors_state(x_interp_to_hres, "input_lres", in_place=False)
            x_proc = x_proc[..., interface.data_indices.data.input[0].full]

        x_hres_proc = interface.pre_processors_state(x_hres_4d, "input_hres", in_place=False)
        x_hres_proc = x_hres_proc[..., interface.data_indices.data.input[1].full]

        x_proc_5d = x_proc[..., None, :, :]
        x_hres_5d = x_hres_proc[..., None, :, :]

        noise_steps = interface.noise_schedule(
            device=x_proc_5d.device,
            dtype=torch.float64,
            nsteps=nsteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )
        _, sampler_args = _split_sampling_args(extra_args)
        states = interface.default_sampler(
            x_proc_5d,
            x_hres_5d,
            noise_steps,
            return_intermediate=True,
            model_comm_group=model_comm_group,
            **sampler_args,
        )

        def _align_interpolated(pred: torch.Tensor, interp: torch.Tensor) -> torch.Tensor:
            aligned = interp
            while aligned.ndim < pred.ndim:
                aligned = aligned.unsqueeze(1)
            return aligned

        y_hat = interface.post_processors_state(states["y_hat"], in_place=False, dataset="output")
        training_cfg = getattr(getattr(interface, "config", None), "training", None)
        if training_cfg is not None and hasattr(training_cfg, "predict_residuals"):
            predicts_residuals = bool(getattr(training_cfg, "predict_residuals"))
        else:
            predicts_residuals = True
        interp_state = interface.interpolate_down(x_in_lres[:, 0, 0, ...], grad_checkpoint=False)
        if predicts_residuals:
            y_hat = y_hat + _align_interpolated(y_hat, interp_state)

        all_intermediates: list[torch.Tensor] = []
        for inter_state in states.get("intermediate_states", []):
            inter_pp = interface.post_processors_state(inter_state, in_place=False, dataset="output")
            if predicts_residuals:
                inter_pp = inter_pp + _align_interpolated(inter_pp, interp_state)
            all_intermediates.append(inter_pp)

        if not all_intermediates:
            raise RuntimeError("No intermediate states were produced by OLD default_sampler.")

        if include_init_state:
            # OLD sampler does not expose pre-denoising initial state explicitly.
            capture_steps_resolved = capture_steps_resolved

        kept_steps: list[np.ndarray] = []
        for s in capture_steps_resolved:
            if 0 <= int(s) < len(all_intermediates):
                kept_steps.append(all_intermediates[int(s)][0, 0, 0].detach().cpu().numpy().astype(np.float32))
        if not kept_steps:
            kept_steps.append(all_intermediates[-1][0, 0, 0].detach().cpu().numpy().astype(np.float32))
            capture_steps_resolved = [len(all_intermediates) - 1]

        final_pred = y_hat[0, 0, 0].detach().cpu().numpy().astype(np.float32)
        inter = np.stack(kept_steps, axis=0)
        return final_pred, inter, np.asarray(capture_steps_resolved, dtype=np.int32), x_interp_state

    pre_processors = getattr(interface, "pre_processors", None)
    if pre_processors is None:
        pre_processors = getattr(interface, "pre_processors_state", None)
    post_processors = getattr(interface, "post_processors", None)
    if post_processors is None:
        post_processors = getattr(interface, "post_processors_state", None)

    predict_kwargs = {
        "x_in": x_in_lres,
        "x_in_hres": x_in_hres,
        "pre_processors": pre_processors,
        "post_processors": post_processors,
        "multi_step": interface.multi_step,
        "model_comm_group": model_comm_group,
    }
    if hasattr(interface, "pre_processors_tendencies"):
        predict_kwargs["pre_processors_tendencies"] = interface.pre_processors_tendencies
    if hasattr(interface, "post_processors_tendencies"):
        predict_kwargs["post_processors_tendencies"] = interface.post_processors_tendencies

    before_sampling_data, grid_shard_shapes = model._before_sampling(**predict_kwargs)
    x_interp = before_sampling_data[0]
    x_hres_proc = before_sampling_data[1]

    noise_args, sampler_args = _split_sampling_args(extra_args)
    sigmas = _build_sigmas(model, x_interp, noise_args)
    total_steps = len(sigmas) - 1
    capture_steps_resolved = resolve_capture_steps(
        total_steps=total_steps,
        explicit_steps=capture_steps,
        capture_max_steps=0,
    )
    capture_steps_expected = list(capture_steps_resolved)
    if include_init_state:
        capture_steps_expected = [-1] + capture_steps_expected

    post_tend = getattr(interface, "post_processors_tendencies", None)
    sampler_name = sampler_args.get("sampler", model.inference_defaults.diffusion_sampler.get("sampler", "heun"))
    if sampler_name == "heun":
        capture_steps_set = set(capture_steps_expected)
        captured_latents: dict[int, torch.Tensor] = {}

        def _capture_step(step_idx: int, latent: torch.Tensor) -> None:
            if step_idx not in capture_steps_set or step_idx in captured_latents:
                return
            captured_latents[step_idx] = latent.detach().to(device="cpu", dtype=torch.float32).clone()

        model.sample(
            x_interp,
            x_hres_proc,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            noise_scheduler_params=noise_args or None,
            sampler_params=sampler_args or None,
            step_callback=_capture_step,
            capture_init_state=include_init_state,
        )

        missing_steps = [step_idx for step_idx in capture_steps_expected if step_idx not in captured_latents]
        if missing_steps:
            raise RuntimeError(f"Missing captured intermediate states for sampling steps: {missing_steps}")

        latent_steps = [captured_latents[step_idx] for step_idx in capture_steps_expected]
    elif sampler_name == "dpmpp_2m":
        shape = (x_interp.shape[0], 1, x_interp.shape[2], x_interp.shape[-2], model.num_output_channels)
        y = torch.randn(shape, device=x_interp.device) * sigmas[0]
        latent_steps = _sample_dpmpp_2m_with_intermediates(
            model=model,
            x_in_interp=x_interp,
            x_in_hres=x_hres_proc,
            y=y,
            sigmas=sigmas,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            capture_steps=set(capture_steps_resolved),
        )
        if include_init_state:
            latent_steps = [y.detach().to(device="cpu", dtype=torch.float32).clone()] + latent_steps
    else:
        raise ValueError(f"Unsupported sampler for intermediate plotting: {sampler_name}")

    if not latent_steps:
        raise RuntimeError("No intermediate states were produced by the sampler.")

    state_steps: list[np.ndarray] = []
    for latent in latent_steps:
        state = model._after_sampling(
            latent.to(device=x_interp.device, dtype=x_interp.dtype),
            post_processors,
            before_sampling_data,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            gather_out=True,
            post_processors_tendencies=post_tend,
        )
        state_steps.append(state[0, 0, 0].detach().cpu().numpy().astype(np.float32))

    final_pred = state_steps[-1]
    x_interp_for_export = x_interp
    if model_comm_group is not None and grid_shard_shapes is not None:
        x_interp_for_export = gather_tensor(
            x_interp_for_export,
            -2,
            apply_shard_shapes(x_interp_for_export, -2, grid_shard_shapes),
            model_comm_group,
        )
    x_interp_state = x_interp_for_export[0, 0, 0].detach().cpu().numpy().astype(np.float32)
    inter = np.stack(state_steps, axis=0)
    return final_pred, inter, np.asarray(capture_steps_expected, dtype=np.int32), x_interp_state


def _build_intermediate_dataset_from_checkpoint(args: argparse.Namespace) -> xr.Dataset:
    global_rank, local_rank, world_size = _get_parallel_info()
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = _resolve_device(requested_device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)

    ckpt_path = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    interface, datamodule, _, _ = _load_objects(
        ckpt_path=ckpt_path,
        device=device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
    )

    data = datamodule.ds_valid.data
    member_id = int(args.member)
    idx = int(args.idx)

    x = np.asarray(data[idx : idx + 1][0])
    x_h = np.asarray(data[idx : idx + 1][1])
    y = np.asarray(data[idx : idx + 1][2])

    max_members = int(x.shape[2])
    if member_id < 0 or member_id >= max_members:
        raise ValueError(f"member={member_id} out of range [0, {max_members - 1}]")

    x = np.transpose(x, (0, 2, 3, 1))
    x_h = np.transpose(x_h, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))

    name_to_idx_in = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index
    x, _ = extract_filtered_input_from_output(x, name_to_idx_in, name_to_idx_out)

    x_l = torch.from_numpy(x[0, member_id]).to(device)[None, None, None, ...]
    x_hres = torch.from_numpy(x_h[0, member_id]).to(device)[None, None, None, ...]

    extra_args = _parse_json(args.extra_args_json)
    capture_steps = _parse_steps_csv(args.capture_steps)
    if not capture_steps:
        total_steps = int(extra_args.get("num_steps", 40))
        capture_steps = resolve_capture_steps(
            total_steps=total_steps,
            explicit_steps=None,
            capture_max_steps=int(args.capture_max_steps),
        )
    with torch.inference_mode():
        final_pred, inter_steps, sampling_step_ids, x_interp_state = _predict_with_intermediates_single_member(
            interface=interface,
            x_in_lres=x_l,
            x_in_hres=x_hres,
            extra_args=extra_args,
            model_comm_group=model_comm_group,
            capture_steps=capture_steps,
            include_init_state=bool(args.include_init_state),
        )

    x_out = x[:, [member_id], :, :]
    y_out = y[:, [member_id], :, :]
    y_pred = final_pred[None, None, :, :]

    ds = build_predictions_dataset(
        x=x_out,
        y=y_out,
        y_pred=y_pred,
        lon_lres=np.asarray(data.longitudes[0]),
        lat_lres=np.asarray(data.latitudes[0]),
        lon_hres=np.asarray(data.longitudes[2]),
        lat_hres=np.asarray(data.latitudes[2]),
        weather_states=list(name_to_idx_out.keys()),
        dates=np.asarray(data.dates[idx : idx + 1]),
        member_ids=[member_id],
    )
    ds["inter_state"] = (
        ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
        inter_steps[None, None, ...],
    )
    if (
        x_interp_state.shape[-1] == len(ds.weather_state)
        and x_interp_state.shape[0] == int(ds.sizes.get("grid_point_hres", -1))
    ):
        ds["x_interp"] = (
            ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
            x_interp_state[None, None, ...],
        )
    ds = ds.assign_coords(sampling_step=sampling_step_ids)
    ds.attrs["intermediate_source"] = "checkpoint"
    ds.attrs["name_ckpt"] = args.name_ckpt
    ds.attrs["idx"] = idx
    ds.attrs["member"] = member_id
    ds.attrs["sampling_config_json"] = args.extra_args_json
    return ds


def _draw_field(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    tc_member_style: bool = False,
    region_box: Sequence[float] | None = None,
    extent_bounds: Sequence[float] | None = None,
):
    if tc_member_style and ccrs is not None:
        levels = np.linspace(float(vmin), float(vmax), 31)
        scatter = ax.tricontourf(
            lon,
            lat,
            field,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=float(vmin),
            vmax=float(vmax),
        )
        ax.tricontour(
            lon,
            lat,
            field,
            levels=levels,
            transform=ccrs.PlateCarree(),
            colors="black",
            linewidths=0.5,
        )
        ax.coastlines()
        ax.grid(color="white", linestyle="--", linewidth=0.5)
        gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
        if extent_bounds is not None and len(extent_bounds) == 4:
            ax.set_extent(
                [float(extent_bounds[0]), float(extent_bounds[1]), float(extent_bounds[2]), float(extent_bounds[3])],
                crs=ccrs.PlateCarree(),
            )
        elif region_box is not None and len(region_box) == 4:
            ax.set_extent(
                [float(region_box[2]), float(region_box[3]), float(region_box[0]), float(region_box[1])],
                crs=ccrs.PlateCarree(),
            )
    else:
        scatter = ax.scatter(
            lon,
            lat,
            c=field,
            s=max(1.0, 75_000 / max(1, len(lon))),
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            rasterized=True,
        )
        if Coastlines is not None:
            Coastlines().plot_continents(ax)
        ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
        ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
        ax.set_aspect("auto", adjustable=None)
        ax.set_box_aspect(1)
    ax.set_title(title)
    return scatter


def _format_panel_title(
    *,
    base: str,
    center_lon: float | None = None,
    center_lat: float | None = None,
    msl_min_hpa: float | None = None,
    wind_max: float | None = None,
) -> str:
    line1 = base
    parts: list[str] = []
    if (center_lon is not None) and (center_lat is not None):
        parts.append(f"c=({center_lon:.1f},{center_lat:.1f})")
    if msl_min_hpa is not None:
        parts.append(f"msl={msl_min_hpa:.1f}hPa")
    if wind_max is not None:
        parts.append(f"w={wind_max:.1f}")
    if not parts:
        return line1
    return f"{line1}\n" + " ".join(parts)


def _human_date_string(value: object) -> str:
    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.datetime64):
            dt = arr.astype("datetime64[s]")
            return np.datetime_as_string(dt, unit="s").replace("T", " ") + " UTC"
        if np.issubdtype(arr.dtype, np.integer):
            iv = int(arr.reshape(()).item())
            # Common case in this workflow: nanoseconds since epoch.
            if abs(iv) > 10_000_000_000:
                dt = np.datetime64(iv, "ns").astype("datetime64[s]")
                return np.datetime_as_string(dt, unit="s").replace("T", " ") + " UTC"
            dt = np.datetime64(iv, "s").astype("datetime64[s]")
            return np.datetime_as_string(dt, unit="s").replace("T", " ") + " UTC"
        return str(arr.reshape(()).item())
    except Exception:
        return str(value)


def _window_extent_within_bounds(
    *,
    center: float,
    half_span: float,
    data_min: float,
    data_max: float,
) -> tuple[float, float]:
    lo = center - half_span
    hi = center + half_span
    target_width = 2.0 * half_span
    if lo < data_min:
        hi += data_min - lo
        lo = data_min
    if hi > data_max:
        lo -= hi - data_max
        hi = data_max
    lo = max(lo, data_min)
    hi = min(hi, data_max)
    if (hi - lo) < target_width:
        missing = target_width - (hi - lo)
        lo = max(data_min, lo - 0.5 * missing)
        hi = min(data_max, hi + 0.5 * missing)
    return float(lo), float(hi)


def _nearest_interp_lres_to_hres(
    *,
    lon_lres: np.ndarray,
    lat_lres: np.ndarray,
    values_lres: np.ndarray,
    lon_hres: np.ndarray,
    lat_hres: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolation from lres points to hres points."""
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.column_stack([lon_lres, lat_lres]))
        _, idx = tree.query(np.column_stack([lon_hres, lat_hres]), k=1)
        return values_lres[idx]
    except Exception:
        out = np.empty_like(lon_hres, dtype=values_lres.dtype)
        for i, (x, y) in enumerate(zip(lon_hres, lat_hres)):
            j = np.argmin((lon_lres - x) ** 2 + (lat_lres - y) ** 2)
            out[i] = values_lres[j]
        return out


def _panel_state_values(
    *,
    ds_member: xr.Dataset,
    panel_kind: str,
    weather_state: str,
    sampling_step: int | None,
) -> np.ndarray | None:
    if weather_state not in ds_member.weather_state.values:
        return None
    if panel_kind == "step":
        if sampling_step is None:
            return None
        return ds_member["inter_state"].sel(sampling_step=int(sampling_step), weather_state=weather_state).values
    if panel_kind == "y_pred":
        return ds_member["y_pred"].sel(weather_state=weather_state).values
    if panel_kind == "y" and "y" in ds_member:
        return ds_member["y"].sel(weather_state=weather_state).values
    return None


def plot_intermediate_trajectory(
    *,
    ds: xr.Dataset,
    weather_state: str,
    sample: int,
    member: int,
    out_path: str,
    region: str = "default",
    max_panels: int = 8,
    sampling_steps: Sequence[int] | None = None,
    independent_color_scales: bool = False,
    show_residuals: bool = True,
    max_cols: int = 0,
    center_track_mode: str = "none",
    center_track_state: str = "",
    center_window_deg: float = 0.0,
    annotate_extremes: bool = False,
    extreme_radius_deg: float = 3.0,
    tc_member_style: bool = False,
    stats_out: str = "",
    panel_scale_mode: str = "percentile",
    dpi: int = 320,
    hide_coordinates: bool = True,
) -> Path:
    if "inter_state" not in ds.variables:
        raise ValueError("Dataset does not contain 'inter_state'.")
    if weather_state not in ds.weather_state.values:
        raise ValueError(f"Unknown weather_state={weather_state}")

    ds_region = get_region_ds(ds, region)
    ds_member = ds_region.sel(sample=sample, ensemble_member=member)
    ds_sel = ds_member.sel(weather_state=weather_state)

    available_steps = [int(v) for v in ds_sel.sampling_step.values]
    if sampling_steps:
        steps = [s for s in [int(v) for v in sampling_steps] if s in set(available_steps)]
    else:
        pos = select_sampling_steps(len(available_steps), max_panels)
        steps = [available_steps[i] for i in pos]
    if not steps:
        raise ValueError("No sampling steps selected.")

    unit_scale = 0.01 if weather_state == "msl" else 1.0
    unit_label = "hPa" if weather_state == "msl" else "native"

    inter_data = ds_sel["inter_state"].sel(sampling_step=steps).values
    pred_data = ds_sel["y_pred"].values * unit_scale
    truth_data = (ds_sel["y"].values * unit_scale) if "y" in ds_sel else None
    base_data = None
    if "x" in ds_sel:
        x_vals = ds_sel["x"].values
        if x_vals.shape == pred_data.shape:
            base_data = x_vals * unit_scale
        elif (
            "grid_point_lres" in ds_sel["x"].dims
            and "lon_lres" in ds_sel
            and "lat_lres" in ds_sel
            and "lon_hres" in ds_sel
            and "lat_hres" in ds_sel
        ):
            base_data = _nearest_interp_lres_to_hres(
                lon_lres=ds_sel["lon_lres"].values,
                lat_lres=ds_sel["lat_lres"].values,
                values_lres=x_vals,
                lon_hres=ds_sel["lon_hres"].values,
                lat_hres=ds_sel["lat_hres"].values,
            ) * unit_scale

    panel_titles: list[str] = [f"step={int(s)}" for s in steps]
    panel_fields: list[np.ndarray] = [
        ds_sel["inter_state"].sel(sampling_step=s).values * unit_scale for s in steps
    ]
    panel_kinds: list[tuple[str, int | None]] = [("step", int(s)) for s in steps]
    panel_titles.append("y_pred")
    panel_fields.append(pred_data)
    panel_kinds.append(("y_pred", None))
    if truth_data is not None:
        panel_titles.append("y")
        panel_fields.append(truth_data)
        panel_kinds.append(("y", None))

    center_track_mode = str(center_track_mode).lower().strip()
    if center_track_mode not in {"none", "min", "max"}:
        raise ValueError("center_track_mode must be one of: none, min, max.")
    center_enabled = center_track_mode != "none"
    center_state = center_track_state.strip() or weather_state
    if center_enabled and center_state not in ds_member.weather_state.values:
        raise ValueError(f"center_track_state={center_state} is not present in weather_state.")

    all_fields = [f.reshape(-1) for f in panel_fields]
    global_vmin = float(np.nanmin(np.concatenate(all_fields)))
    global_vmax = float(np.nanmax(np.concatenate(all_fields)))

    residual_enabled = bool(show_residuals and (base_data is not None))
    residual_fields = [field - base_data for field in panel_fields] if residual_enabled else []
    if residual_enabled:
        residual_concat = np.concatenate([rf.reshape(-1) for rf in residual_fields])

    n_panels = len(panel_fields)
    ncols = n_panels if max_cols <= 0 else min(max_cols, n_panels)
    nfield_rows = (n_panels + ncols - 1) // ncols
    nrows = nfield_rows * (2 if residual_enabled else 1)
    subplot_kw = {"projection": ccrs.PlateCarree()} if (tc_member_style and ccrs is not None) else {}
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.9 * nrows),
        squeeze=False,
        subplot_kw=subplot_kw,
    )

    lon = ds_sel["lon_hres"].values
    lat = ds_sel["lat_hres"].values

    panel_scale_mode = str(panel_scale_mode).strip().lower()
    if panel_scale_mode not in {"percentile", "minmax"}:
        raise ValueError("panel_scale_mode must be one of: percentile, minmax.")

    if independent_color_scales:
        row0_vmin, row0_vmax = global_vmin, global_vmax
    else:
        row0_vmin = float(np.nanpercentile(np.concatenate([f.reshape(-1) for f in panel_fields]), 1))
        row0_vmax = float(np.nanpercentile(np.concatenate([f.reshape(-1) for f in panel_fields]), 99))
        if not np.isfinite(row0_vmin) or not np.isfinite(row0_vmax) or row0_vmin == row0_vmax:
            row0_vmin, row0_vmax = global_vmin, global_vmax

    used_axes = set()
    panel_masks: list[np.ndarray] = []
    for i, field in enumerate(panel_fields):
        mask = np.ones_like(lon, dtype=bool)
        center_lon: float | None = None
        center_lat: float | None = None
        msl_min_hpa: float | None = None
        wind_max: float | None = None
        if center_enabled:
            panel_kind, panel_step = panel_kinds[i]
            center_field = _panel_state_values(
                ds_member=ds_member,
                panel_kind=panel_kind,
                weather_state=center_state,
                sampling_step=panel_step,
            )
            if center_field is not None:
                center_idx = int(np.nanargmin(center_field)) if center_track_mode == "min" else int(np.nanargmax(center_field))
                center_lon = float(lon[center_idx])
                center_lat = float(lat[center_idx])
                if center_window_deg > 0:
                    local_mask = (
                        (np.abs(lon - center_lon) <= center_window_deg)
                        & (np.abs(lat - center_lat) <= center_window_deg)
                    )
                    if (not tc_member_style) and int(np.count_nonzero(local_mask)) >= 20:
                        mask = local_mask

                if annotate_extremes:
                    radius = max(0.1, float(extreme_radius_deg))
                    dist2 = (lon - center_lon) ** 2 + (lat - center_lat) ** 2
                    ext_mask = dist2 <= radius**2
                    if int(np.count_nonzero(ext_mask)) > 0:
                        msl_vals = _panel_state_values(
                            ds_member=ds_member,
                            panel_kind=panel_kind,
                            weather_state="msl",
                            sampling_step=panel_step,
                        )
                        u_vals = _panel_state_values(
                            ds_member=ds_member,
                            panel_kind=panel_kind,
                            weather_state="10u",
                            sampling_step=panel_step,
                        )
                        v_vals = _panel_state_values(
                            ds_member=ds_member,
                            panel_kind=panel_kind,
                            weather_state="10v",
                            sampling_step=panel_step,
                        )
                        if msl_vals is not None:
                            msl_min_hpa = float(np.nanmin(msl_vals[ext_mask]) * 0.01)
                        if (u_vals is not None) and (v_vals is not None):
                            wind = np.sqrt(np.asarray(u_vals) ** 2 + np.asarray(v_vals) ** 2)
                            wind_max = float(np.nanmax(wind[ext_mask]))

        panel_masks.append(mask)
        field_plot = field[mask]
        lon_plot = lon[mask]
        lat_plot = lat[mask]
        region_box = ds_region.attrs.get("region")
        if (center_enabled and (center_window_deg > 0) and (center_lon is not None) and (center_lat is not None)):
            # Keep a constant panel span around tracked center so all frames are visually comparable.
            span = float(center_window_deg)
            lon_lo, lon_hi = _window_extent_within_bounds(
                center=center_lon,
                half_span=span,
                data_min=float(np.nanmin(lon)),
                data_max=float(np.nanmax(lon)),
            )
            lat_lo, lat_hi = _window_extent_within_bounds(
                center=center_lat,
                half_span=span,
                data_min=float(np.nanmin(lat)),
                data_max=float(np.nanmax(lat)),
            )
            extent_bounds = [lon_lo, lon_hi, lat_lo, lat_hi]
        elif region_box is not None and len(region_box) == 4:
            extent_bounds = [float(region_box[2]), float(region_box[3]), float(region_box[0]), float(region_box[1])]
        else:
            lon_min = float(np.nanmin(lon))
            lon_max = float(np.nanmax(lon))
            lat_min = float(np.nanmin(lat))
            lat_max = float(np.nanmax(lat))
            extent_bounds = [lon_min, lon_max, lat_min, lat_max]
        rr = i // ncols
        cc = i % ncols
        if independent_color_scales:
            if panel_scale_mode == "minmax":
                vmin = float(np.nanmin(field))
                vmax = float(np.nanmax(field))
            else:
                vmin = float(np.nanpercentile(field, 1))
                vmax = float(np.nanpercentile(field, 99))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = row0_vmin, row0_vmax
        else:
            vmin, vmax = row0_vmin, row0_vmax
        mappable = _draw_field(
            axs[rr, cc],
            lon_plot,
            lat_plot,
            field_plot,
            title=_format_panel_title(
                base=panel_titles[i],
                center_lon=center_lon,
                center_lat=center_lat,
                msl_min_hpa=msl_min_hpa,
                wind_max=wind_max,
            ),
            vmin=vmin,
            vmax=vmax,
            tc_member_style=bool(tc_member_style),
            region_box=ds_region.attrs.get("region"),
            extent_bounds=extent_bounds,
        )
        used_axes.add((rr, cc))
        if independent_color_scales:
            cbar = fig.colorbar(
                mappable,
                ax=axs[rr, cc],
                orientation="horizontal",
                shrink=0.62,
                fraction=0.045,
                pad=0.03,
            )
            cbar.outline.set_edgecolor("black")
            cbar.outline.set_linewidth(1.0)
            cbar.ax.tick_params(labelsize=8)
            cbar.locator = ticker.MaxNLocator(nbins=4)
            cbar.update_ticks()

    if not independent_color_scales and used_axes:
        top_axes = [axs[r, c] for (r, c) in sorted(used_axes) if r < nfield_rows]
        if top_axes:
            cbar_top = fig.colorbar(
                plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=row0_vmin, vmax=row0_vmax)),
                ax=top_axes,
                orientation="horizontal",
                fraction=0.025,
                pad=0.04,
                aspect=40,
            )
            cbar_top.outline.set_edgecolor("black")
            cbar_top.outline.set_linewidth(1.0)
            cbar_top.ax.tick_params(labelsize=9)
            cbar_top.locator = ticker.MaxNLocator(nbins=5)
            cbar_top.update_ticks()
            cbar_top.set_label(f"{weather_state} ({unit_label})", fontsize=9)

    if residual_enabled:
        if independent_color_scales:
            row1_vmin, row1_vmax = float(np.nanmin(residual_concat)), float(np.nanmax(residual_concat))
        else:
            row1_vmin = float(np.nanpercentile(residual_concat, 1))
            row1_vmax = float(np.nanpercentile(residual_concat, 99))
            if not np.isfinite(row1_vmin) or not np.isfinite(row1_vmax) or row1_vmin == row1_vmax:
                row1_vmin, row1_vmax = float(np.nanmin(residual_concat)), float(np.nanmax(residual_concat))

        for i, rfield in enumerate(residual_fields):
            mask = panel_masks[i] if i < len(panel_masks) else np.ones_like(rfield, dtype=bool)
            rfield_plot = rfield[mask]
            lon_plot = lon[mask]
            lat_plot = lat[mask]
            region_box = ds_region.attrs.get("region")
            if region_box is not None and len(region_box) == 4:
                extent_bounds = [float(region_box[2]), float(region_box[3]), float(region_box[0]), float(region_box[1])]
            else:
                lon_min = float(np.nanmin(lon))
                lon_max = float(np.nanmax(lon))
                lat_min = float(np.nanmin(lat))
                lat_max = float(np.nanmax(lat))
                extent_bounds = [lon_min, lon_max, lat_min, lat_max]
            rr = (i // ncols) + nfield_rows
            cc = i % ncols
            if independent_color_scales:
                if panel_scale_mode == "minmax":
                    rvmin = float(np.nanmin(rfield))
                    rvmax = float(np.nanmax(rfield))
                else:
                    rvmin = float(np.nanpercentile(rfield, 1))
                    rvmax = float(np.nanpercentile(rfield, 99))
                if not np.isfinite(rvmin) or not np.isfinite(rvmax) or rvmin == rvmax:
                    rvmin, rvmax = row1_vmin, row1_vmax
            else:
                rvmin, rvmax = row1_vmin, row1_vmax
            rmappable = _draw_field(
                axs[rr, cc],
                lon_plot,
                lat_plot,
                rfield_plot,
                title=f"{panel_titles[i]} - x_lres_interp",
                vmin=rvmin,
                vmax=rvmax,
                tc_member_style=bool(tc_member_style),
                region_box=ds_region.attrs.get("region"),
                extent_bounds=extent_bounds,
            )
            used_axes.add((rr, cc))
            rmappable.set_cmap("viridis")
            if independent_color_scales:
                cbar = fig.colorbar(
                    rmappable,
                    ax=axs[rr, cc],
                    orientation="horizontal",
                    shrink=0.62,
                    fraction=0.045,
                    pad=0.03,
                )
                cbar.outline.set_edgecolor("black")
                cbar.outline.set_linewidth(1.0)
                cbar.ax.tick_params(labelsize=8)
                cbar.locator = ticker.MaxNLocator(nbins=4)
                cbar.update_ticks()

        if not independent_color_scales:
            bottom_axes = [axs[r, c] for (r, c) in sorted(used_axes) if r >= nfield_rows]
            if bottom_axes:
                cbar_bottom = fig.colorbar(
                    plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=row1_vmin, vmax=row1_vmax)),
                    ax=bottom_axes,
                    orientation="horizontal",
                    fraction=0.025,
                    pad=0.04,
                    aspect=40,
                )
                cbar_bottom.outline.set_edgecolor("black")
                cbar_bottom.outline.set_linewidth(1.0)
                cbar_bottom.ax.tick_params(labelsize=9)
                cbar_bottom.locator = ticker.MaxNLocator(nbins=5)
                cbar_bottom.update_ticks()
                cbar_bottom.set_label(f"{weather_state} - x_lres_interp ({unit_label})", fontsize=9)

    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if (r, c) not in used_axes:
                ax.axis("off")
                continue
            if not tc_member_style:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            if hide_coordinates:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                # Keep only outer tick labels to reduce repetitive clutter.
                is_bottom_row = r == (nrows - 1)
                is_left_col = c == 0
                ax.tick_params(axis="both", which="major", labelsize=9)
                ax.tick_params(axis="x", labelbottom=is_bottom_row)
                ax.tick_params(axis="y", labelleft=is_left_col)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth(2)
            ax.set_title(ax.get_title(), fontsize=9, pad=6)

    date_str = "na"
    if "date" in ds_member:
        try:
            date_str = _human_date_string(np.asarray(ds_member["date"].values).item())
        except Exception:
            date_str = _human_date_string(ds_member["date"].values)
    fig.suptitle(
        f"Intermediate diffusion states | {weather_state} | region={region} | sample={sample} | member={member} | date={date_str} | ref=x_lres_interp",
        y=0.995,
        fontsize=11,
    )
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.045, top=0.92, wspace=0.24, hspace=0.32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=max(120, int(dpi)), bbox_inches="tight")
    plt.close(fig)

    if stats_out:
        stats: dict[str, object] = {
            "weather_state": weather_state,
            "unit_label": unit_label,
            "region": region,
            "sample": int(sample),
            "member": int(member),
            "panel_stats": [],
            "reference_stats": {},
            "final_step_checks": {},
        }

        if base_data is not None:
            stats["reference_stats"] = {
                "x_lres_interp_min": float(np.nanmin(base_data)),
                "x_lres_interp_max": float(np.nanmax(base_data)),
                "x_lres_interp_mean": float(np.nanmean(base_data)),
                "x_lres_interp_std": float(np.nanstd(base_data)),
            }
        if "x_interp" in ds_sel:
            x_interp_vals = ds_sel["x_interp"].values
            stats["reference_stats"] = dict(stats["reference_stats"]) if stats["reference_stats"] else {}
            stats["reference_stats"]["x_interp_internal_min"] = float(np.nanmin(x_interp_vals))
            stats["reference_stats"]["x_interp_internal_max"] = float(np.nanmax(x_interp_vals))
            stats["reference_stats"]["x_interp_internal_mean"] = float(np.nanmean(x_interp_vals))
            stats["reference_stats"]["x_interp_internal_std"] = float(np.nanstd(x_interp_vals))

        panel_stats = []
        for i, field in enumerate(panel_fields):
            kind, step = panel_kinds[i]
            item = {
                "panel_title": panel_titles[i],
                "panel_kind": kind,
                "sampling_step": None if step is None else int(step),
                "full_min": float(np.nanmin(field)),
                "full_max": float(np.nanmax(field)),
                "full_mean": float(np.nanmean(field)),
                "full_std": float(np.nanstd(field)),
            }
            if residual_enabled and i < len(residual_fields):
                rfield = residual_fields[i]
                item["full_minus_x_lres_interp_min"] = float(np.nanmin(rfield))
                item["full_minus_x_lres_interp_max"] = float(np.nanmax(rfield))
                item["full_minus_x_lres_interp_mean"] = float(np.nanmean(rfield))
                item["full_minus_x_lres_interp_std"] = float(np.nanstd(rfield))
            panel_stats.append(item)
        stats["panel_stats"] = panel_stats

        if steps:
            last_step = int(steps[-1])
            last_field = ds_sel["inter_state"].sel(sampling_step=last_step).values * unit_scale
            final_checks = {
                "last_sampling_step": last_step,
                "last_step_vs_y_pred_rmse": float(np.sqrt(np.nanmean((last_field - pred_data) ** 2))),
                "last_step_vs_y_pred_max_abs": float(np.nanmax(np.abs(last_field - pred_data))),
            }
            if truth_data is not None:
                final_checks["y_pred_vs_y_rmse"] = float(np.sqrt(np.nanmean((pred_data - truth_data) ** 2)))
                final_checks["y_pred_vs_y_max_abs"] = float(np.nanmax(np.abs(pred_data - truth_data)))
            stats["final_step_checks"] = final_checks

        stats_path = Path(stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return out


def _load_dataset(args: argparse.Namespace) -> xr.Dataset:
    if args.cmd == "checkpoint":
        ds = _build_intermediate_dataset_from_checkpoint(args)
        if args.save_intermediate_nc:
            out_nc = Path(args.save_intermediate_nc)
            out_nc.parent.mkdir(parents=True, exist_ok=True)
            if out_nc.exists():
                out_nc.unlink()
            ds.to_netcdf(out_nc)
            print(f"Saved intermediate dataset: {out_nc}")
        return ds

    ds = xr.open_dataset(args.predictions_nc)
    return ds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot intermediate diffusion sampling states from checkpoint or dataset.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ckpt = sub.add_parser("checkpoint", help="Generate and plot intermediates from a checkpoint.")
    p_ckpt.add_argument("--name-ckpt", required=True)
    p_ckpt.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    p_ckpt.add_argument("--device", default="cuda")
    p_ckpt.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_ckpt.add_argument("--validation-frequency", default="50h")
    p_ckpt.add_argument("--idx", type=int, default=0)
    p_ckpt.add_argument("--member", type=int, default=0)
    p_ckpt.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_ckpt.add_argument("--save-intermediate-nc", default="")
    p_ckpt.add_argument(
        "--capture-steps",
        default="",
        help="Comma-separated diffusion step ids to retain in inter_state.",
    )
    p_ckpt.add_argument(
        "--capture-max-steps",
        type=int,
        default=8,
        help="If --capture-steps is empty, keep at most this many steps (0 keeps all).",
    )
    p_ckpt.add_argument(
        "--include-init-state",
        action="store_true",
        help="Include pre-denoising initialization state as sampling_step=-1.",
    )

    p_ds = sub.add_parser("dataset", help="Plot intermediates from an existing dataset with inter_state.")
    p_ds.add_argument("--predictions-nc", required=True)

    for p in (p_ckpt, p_ds):
        p.add_argument("--sample", type=int, default=0)
        p.add_argument("--weather-state", required=True)
        p.add_argument("--region", default="default")
        p.add_argument("--max-panels", type=int, default=8)
        p.add_argument("--steps", default="", help="Comma-separated sampling_step indices.")
        p.add_argument(
            "--independent-color-scales",
            action="store_true",
            help="Use independent color scales per panel instead of one shared scale.",
        )
        p.add_argument(
            "--no-residuals",
            action="store_true",
            help="Disable residual row (field - x_interp).",
        )
        p.add_argument(
            "--max-cols",
            type=int,
            default=0,
            help="Maximum number of panels per row (0 keeps all panels in one row).",
        )
        p.add_argument(
            "--center-track-mode",
            choices=["none", "min", "max"],
            default="none",
            help="Track center on a reference weather_state using min or max value.",
        )
        p.add_argument(
            "--center-track-state",
            default="",
            help="Weather state used for center tracking (default: current --weather-state).",
        )
        p.add_argument(
            "--center-window-deg",
            type=float,
            default=0.0,
            help="If >0, crop each panel to a +/-degree window around tracked center.",
        )
        p.add_argument(
            "--annotate-extremes",
            action="store_true",
            help="Annotate center-local MSLP minimum and wind maximum near tracked center.",
        )
        p.add_argument(
            "--extreme-radius-deg",
            type=float,
            default=3.0,
            help="Radius in degrees for center-local extreme diagnostics.",
        )
        p.add_argument(
            "--tc-member-style",
            action="store_true",
            help="Use TC member map style (PlateCarree + contour overlays + gridline labels).",
        )
        p.add_argument(
            "--also-region-style-out",
            default="",
            help="Optional second output path rendered in region_plotting-like style (non-TC style).",
        )
        p.add_argument(
            "--stats-out",
            default="",
            help="Optional JSON path to save panel/reference/final-step statistics.",
        )
        p.add_argument(
            "--panel-scale-mode",
            choices=["percentile", "minmax"],
            default="percentile",
            help="Scale mode when --independent-color-scales is enabled.",
        )
        p.add_argument("--dpi", type=int, default=320, help="Output DPI for raster export.")
        p.add_argument(
            "--show-coordinates",
            action="store_true",
            help="Show longitude/latitude ticks and labels (default hides them for readability).",
        )
        p.add_argument("--out", required=True, help="Output image path (png/pdf).")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = _load_dataset(args)
    steps = _parse_steps_csv(args.steps)

    out = plot_intermediate_trajectory(
        ds=ds,
        weather_state=args.weather_state,
        sample=int(args.sample),
        member=int(getattr(args, "member", 0)),
        out_path=args.out,
        region=args.region,
        max_panels=int(args.max_panels),
        sampling_steps=steps,
        independent_color_scales=bool(args.independent_color_scales),
        show_residuals=not bool(args.no_residuals),
        max_cols=int(args.max_cols),
        center_track_mode=str(args.center_track_mode),
        center_track_state=str(args.center_track_state),
        center_window_deg=float(args.center_window_deg),
        annotate_extremes=bool(args.annotate_extremes),
        extreme_radius_deg=float(args.extreme_radius_deg),
        tc_member_style=bool(args.tc_member_style),
        stats_out=str(getattr(args, "stats_out", "")),
        panel_scale_mode=str(getattr(args, "panel_scale_mode", "percentile")),
        dpi=int(getattr(args, "dpi", 320)),
        hide_coordinates=not bool(getattr(args, "show_coordinates", False)),
    )
    print(f"Saved intermediate trajectory plot: {out}")
    if str(getattr(args, "also_region_style_out", "")).strip():
        out_region = plot_intermediate_trajectory(
            ds=ds,
            weather_state=args.weather_state,
            sample=int(args.sample),
            member=int(getattr(args, "member", 0)),
            out_path=str(args.also_region_style_out),
            region=args.region,
            max_panels=int(args.max_panels),
            sampling_steps=steps,
            independent_color_scales=bool(args.independent_color_scales),
            show_residuals=not bool(args.no_residuals),
            max_cols=int(args.max_cols),
            center_track_mode=str(args.center_track_mode),
            center_track_state=str(args.center_track_state),
            center_window_deg=float(args.center_window_deg),
            annotate_extremes=bool(args.annotate_extremes),
            extreme_radius_deg=float(args.extreme_radius_deg),
            tc_member_style=False,
            panel_scale_mode=str(getattr(args, "panel_scale_mode", "percentile")),
            dpi=int(getattr(args, "dpi", 320)),
            hide_coordinates=not bool(getattr(args, "show_coordinates", False)),
        )
        print(f"Saved region-style intermediate trajectory plot: {out_region}")


if __name__ == "__main__":
    main()
