"""Rectified-flow ODE sampler for flow-matching super-resolution.

The flow-matching counterpart of ``stochastic_sampler`` (EDM Heun). The network
predicts the velocity ``v_theta(x_t, t)`` of the linear probability path

    x_t = (1 - t) * x_data + t * eps,    t in [0, 1]

so sampling integrates the probability-flow ODE ``dx/dt = v_theta`` from pure
noise at ``t = t_max`` (=1) down to data at ``t = t_min`` (=0):

    Euler:  x <- x + (t_next - t_cur) * v(x, t_cur)
    Heun :  x_euler = x + dt * v(x, t_cur)
            x       = x + dt * 0.5 * (v(x, t_cur) + v(x_euler, t_next))

Conditioning (``mean_hr`` / ``static_channels`` / spatial ``date_embedding``) is
assembled into ``x_lr`` exactly as in ``stochastic_sampler`` so the two samplers
are drop-in interchangeable behind ``GeneratorBase.initialize_sampler``. For the
DiT the date embedding is a global AdaLN condition vector passed via ``model_args``
(``{"condition": ...}``), not a spatial channel, so ``date_embedding`` stays None.

The call/return signature matches ``stochastic_sampler`` (same kwargs are forwarded
by ``diffusion_step``); EDM-only kwargs (``S_churn`` etc.) are absent by design.
"""

from typing import Callable, Optional

import torch
from torch import Tensor

from hirad.utils.patching import GridPatching2D


def _sync_t() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import time

    return time.time()


def flow_matching_sampler(
    net: torch.nn.Module,
    latents: torch.Tensor,
    img_lr: torch.Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[torch.Tensor] = None,
    lead_time_label: Optional[torch.Tensor] = None,
    static_channels: Optional[torch.Tensor] = None,
    date_embedding: Optional[torch.Tensor] = None,
    num_steps: int = 18,
    t_min: float = 0.0,
    t_max: float = 1.0,
    heun: bool = True,
    use_apex_gn: bool = False,
    _timings: Optional[dict] = None,
    model_args: Optional[dict] = None,
) -> torch.Tensor:
    """Integrate the rectified-flow ODE from noise (t=t_max) to data (t=t_min).

    Parameters
    ----------
    net : torch.nn.Module
        Flow-matching network with signature ``net(x, x_lr, t, condition=...)``
        returning the velocity of shape (B, C_out, H, W).
    latents : torch.Tensor
        Initial pure noise ``~ N(0, I)`` of shape (B, C_out, H, W); the ODE state
        at ``t = t_max``.
    img_lr : torch.Tensor
        Low-resolution conditioning of shape (B, C_lr, H, W).
    num_steps : int, optional
        Number of ODE integration steps, by default 18.
    t_min, t_max : float, optional
        Integration endpoints, by default 0.0 and 1.0. ``t_max`` should be 1.0 so
        that ``latents`` is a valid pure-noise initial state.
    heun : bool, optional
        Use the 2nd-order Heun corrector (except on the final step), by default True.
    model_args : Optional[dict], optional
        Extra kwargs forwarded to ``net`` (e.g. ``{"condition": date_embedding}``).

    Returns
    -------
    torch.Tensor
        The generated sample at ``t = t_min`` (the data end), shape as ``latents``.
    """
    model_args = dict(model_args or {})

    if patching is not None:
        # The DiT flow-matching path does not use positional-embedding patching
        # (the DiT has no embedding_selector); plain full-domain inference only.
        raise NotImplementedError(
            "flow_matching_sampler does not support patched generation."
        )

    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

    _t = _sync_t if _timings is not None else (lambda: 0.0)
    _t_preproc_start = _t()

    batch_size = img_lr.shape[0]

    # conditioning = [mean_hr, img_lr, static, (spatial date)]  -- mirrors stochastic_sampler
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)

    if static_channels is not None:
        if static_channels.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"static_channels and img_lr must have the same height and width, "
                f"but found {static_channels.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((x_lr, static_channels.expand(batch_size, -1, -1, -1)), dim=1)

    # Spatial date channels (unused for the DiT, which passes date as a condition
    # vector via model_args; kept for parity with stochastic_sampler).
    if date_embedding is not None:
        date_spatial = date_embedding[:, :, None, None].expand(
            x_lr.shape[0], date_embedding.shape[1], *x_lr.shape[2:]
        )
        if use_apex_gn:
            date_spatial = date_spatial.to(x_lr.dtype, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            date_spatial = date_spatial.to(x_lr.dtype, non_blocking=True).contiguous()
        x_lr = torch.cat((x_lr, date_spatial), dim=1)

    x_lr = x_lr.to(latents.device)

    if lead_time_label is not None:
        model_args["lead_time_label"] = lead_time_label

    # Time discretization: t_max (noise) -> t_min (data), uniform.
    t_steps = torch.linspace(
        t_max, t_min, num_steps + 1, dtype=torch.float64, device=latents.device
    )

    def _v(x_state: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        t_batch = torch.full(
            (x_state.shape[0],), float(t_val), device=x_state.device, dtype=torch.float64
        )
        return net(x_state, x_lr, t_batch, **model_args).to(torch.float64)

    _t_preproc_end = _t()
    _t_net_forward = 0.0
    _n_net_forward = 0
    _t_loop_start = _t()

    # Initial pure-noise state at t = t_max (t_max should be 1.0 -> unit gaussian).
    x = latents.to(torch.float64)

    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        dt = t_next - t_cur  # negative (integrating toward data)

        _tn0 = _t()
        v_cur = _v(x, t_cur)
        _t_net_forward += _t() - _tn0
        _n_net_forward += 1

        if heun and i < num_steps - 1:
            x_euler = x + dt * v_cur
            _tn0 = _t()
            v_next = _v(x_euler, t_next)
            _t_net_forward += _t() - _tn0
            _n_net_forward += 1
            x = x + dt * 0.5 * (v_cur + v_next)
        else:
            x = x + dt * v_cur

    _t_loop_end = _t()

    if _timings is not None:
        _timings["fm_preproc"] = _timings.get("fm_preproc", 0.0) + (
            _t_preproc_end - _t_preproc_start
        )
        _timings["fm_net_forward"] = _timings.get("fm_net_forward", 0.0) + _t_net_forward
        _timings["fm_net_forward_count"] = (
            _timings.get("fm_net_forward_count", 0) + _n_net_forward
        )
        _timings["fm_loop_total"] = _timings.get("fm_loop_total", 0.0) + (
            _t_loop_end - _t_loop_start
        )

    return x.to(latents.dtype)
