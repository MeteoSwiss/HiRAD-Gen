from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import torch


DEFAULT_LANE = "o96_o320"
DEFAULT_SIGMA_MIN = 0.03
DEFAULT_RHO = 7.0
DEFAULT_SAMPLER = "heun"
DEFAULT_S_CHURN = 2.5
DEFAULT_S_MIN = 0.75
DEFAULT_S_NOISE = 1.05
DEFAULT_TRANSITION_CONTROLS = (1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_STAGE0_SIGMA_MAX_VALUES = (1000.0, 10000.0, 100000.0)
DEFAULT_STAGE0_NUM_STEPS_HIGH_VALUES = (3, 5, 7)
DEFAULT_STAGE1_NUM_STEPS_LOW_VALUES = (12, 16, 20)
CONTROL_CANDIDATES = (
    {
        "candidate_key": "control_piecewise30_reference",
        "label": "piecewise30_reference",
        "extra_args": {
            "schedule_type": "experimental_piecewise",
            "schedule_kind": "piecewise30",
            "num_steps": 30,
            "sigma_max": 1000.0,
            "sigma_transition": 10.0,
            "sigma_min": DEFAULT_SIGMA_MIN,
            "high_schedule_type": "exponential",
            "low_schedule_type": "karras",
            "num_steps_high": 10,
            "num_steps_low": 20,
            "rho": DEFAULT_RHO,
            "sampler": DEFAULT_SAMPLER,
            "S_churn": DEFAULT_S_CHURN,
            "S_min": DEFAULT_S_MIN,
            "S_max": 1000.0,
            "S_noise": DEFAULT_S_NOISE,
        },
    },
    {
        "candidate_key": "control_karras40_sigmax1000",
        "label": "karras40_sigmax1000",
        "extra_args": {
            "schedule_type": "karras",
            "schedule_kind": "karras40",
            "num_steps": 40,
            "sigma_max": 1000.0,
            "sigma_min": DEFAULT_SIGMA_MIN,
            "rho": DEFAULT_RHO,
            "sampler": DEFAULT_SAMPLER,
            "S_churn": DEFAULT_S_CHURN,
            "S_min": DEFAULT_S_MIN,
            "S_max": 1000.0,
            "S_noise": DEFAULT_S_NOISE,
        },
    },
    {
        "candidate_key": "control_karras80_sigmax100000",
        "label": "karras80_sigmax100k",
        "extra_args": {
            "schedule_type": "karras",
            "schedule_kind": "karras80",
            "num_steps": 80,
            "sigma_max": 100000.0,
            "sigma_min": DEFAULT_SIGMA_MIN,
            "rho": DEFAULT_RHO,
            "sampler": DEFAULT_SAMPLER,
            "S_churn": DEFAULT_S_CHURN,
            "S_min": DEFAULT_S_MIN,
            "S_max": 100000.0,
            "S_noise": DEFAULT_S_NOISE,
        },
    },
)


@dataclass(frozen=True)
class CheckpointProfile:
    checkpoint_id: str
    checkpoint_short: str
    checkpoint_path: str
    baseline_slug: str
    stack_flavor: str = "new"
    lane: str = DEFAULT_LANE

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointProfile":
        checkpoint_id = str(raw["checkpoint_id"])
        checkpoint_short = str(raw.get("checkpoint_short") or checkpoint_id[:8])
        baseline_slug = str(
            raw.get("baseline_slug")
            or f"manual-{checkpoint_short}-{raw.get('stack_flavor', 'new')}-piecewise30-t10-h10-l20-sigma1k"
        )
        return cls(
            checkpoint_id=checkpoint_id,
            checkpoint_short=checkpoint_short,
            checkpoint_path=str(raw["checkpoint_path"]),
            baseline_slug=baseline_slug,
            stack_flavor=str(raw.get("stack_flavor", "new")),
            lane=str(raw.get("lane", DEFAULT_LANE)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerCandidate:
    study: str
    stage: str
    checkpoint_id: str
    checkpoint_short: str
    checkpoint_path: str
    stack_flavor: str
    lane: str
    baseline_slug: str
    candidate_key: str
    candidate_label: str
    family: str
    run_id: str
    total_steps: int
    extra_args: dict[str, Any]
    sigma_schedule: list[float]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SchedulerCandidate":
        return cls(
            study=str(raw.get("study", "o96_o320_piecewise_scheduler_search")),
            stage=str(raw["stage"]),
            checkpoint_id=str(raw["checkpoint_id"]),
            checkpoint_short=str(raw["checkpoint_short"]),
            checkpoint_path=str(raw["checkpoint_path"]),
            stack_flavor=str(raw.get("stack_flavor", "new")),
            lane=str(raw.get("lane", DEFAULT_LANE)),
            baseline_slug=str(raw.get("baseline_slug", "")),
            candidate_key=str(raw["candidate_key"]),
            candidate_label=str(raw.get("candidate_label", raw["candidate_key"])),
            family=str(raw.get("family", "search")),
            run_id=str(raw["run_id"]),
            total_steps=int(raw["total_steps"]),
            extra_args=dict(raw["extra_args"]),
            sigma_schedule=[float(x) for x in raw.get("sigma_schedule", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _format_token(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        integer = int(value)
        if abs(integer) >= 1000 and integer % 1000 == 0:
            return f"{integer // 1000}k"
        return str(integer)
    return str(value).replace(".", "p")


def _default_run_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def load_checkpoint_profiles(path: str | Path) -> list[CheckpointProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["checkpoints"] if isinstance(raw, dict) and "checkpoints" in raw else raw
    return [CheckpointProfile.from_dict(dict(item)) for item in rows]


def _load_candidate_payload(path: str | Path) -> SchedulerCandidate:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "selected_candidate" in raw:
            raw = raw["selected_candidate"]
        elif "candidate" in raw:
            raw = raw["candidate"]
        elif "candidates" in raw:
            candidates = raw["candidates"]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one candidate in best-candidate JSON.")
            raw = candidates[0]
    if not isinstance(raw, dict):
        raise ValueError("Best-candidate JSON must decode to an object.")
    return SchedulerCandidate.from_dict(raw)


def _get_noise_scheduler_classes():
    from anemoi.models.samplers.diffusion_samplers import NOISE_SCHEDULERS

    return NOISE_SCHEDULERS


def _load_ecmwf_spectra_metrics(spectra_dir: Path, *, reference_root: Path | None = None) -> dict[str, Any]:
    from eval.jobs.scoreboard_metrics import load_spectra_metrics

    return load_spectra_metrics(spectra_dir, reference_root=reference_root)


def materialize_sigma_schedule(extra_args: dict[str, Any]) -> list[float]:
    schedule_type = str(extra_args["schedule_type"])
    scheduler_cls = _get_noise_scheduler_classes()[schedule_type]
    scheduler_kwargs = {
        "sigma_max": float(extra_args["sigma_max"]),
        "sigma_min": float(extra_args.get("sigma_min", DEFAULT_SIGMA_MIN)),
        "num_steps": int(extra_args["num_steps"]),
    }
    if "rho" in extra_args:
        scheduler_kwargs["rho"] = float(extra_args["rho"])
    for optional_key in (
        "sigma_transition",
        "high_schedule_type",
        "low_schedule_type",
        "num_steps_high",
        "num_steps_low",
        "rho_high",
        "rho_low",
    ):
        if optional_key in extra_args:
            scheduler_kwargs[optional_key] = extra_args[optional_key]
    scheduler = scheduler_cls(**scheduler_kwargs)
    schedule = scheduler.get_schedule(dtype_compute=torch.float64)
    return [float(value) for value in schedule.detach().cpu().tolist()]


def _build_run_id(profile: CheckpointProfile, run_date: str, candidate_key: str) -> str:
    return (
        f"manual_{profile.checkpoint_short}_{profile.stack_flavor}_{profile.lane}_{run_date}_{candidate_key}"
    )


def _make_piecewise_extra_args(
    *,
    sigma_max: float,
    sigma_transition: float,
    num_steps_high: int,
    num_steps_low: int,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    rho: float = DEFAULT_RHO,
    sampler: str = DEFAULT_SAMPLER,
    s_churn: float = DEFAULT_S_CHURN,
    s_min: float = DEFAULT_S_MIN,
    s_noise: float = DEFAULT_S_NOISE,
    stage_tag: str,
) -> dict[str, Any]:
    total_steps = int(num_steps_high) + int(num_steps_low)
    return {
        "schedule_type": "experimental_piecewise",
        "schedule_kind": f"{stage_tag}_piecewise",
        "num_steps": total_steps,
        "sigma_max": float(sigma_max),
        "sigma_transition": float(sigma_transition),
        "sigma_min": float(sigma_min),
        "high_schedule_type": "exponential",
        "low_schedule_type": "karras",
        "num_steps_high": int(num_steps_high),
        "num_steps_low": int(num_steps_low),
        "rho": float(rho),
        "sampler": sampler,
        "S_churn": float(s_churn),
        "S_min": float(s_min),
        "S_max": float(sigma_max),
        "S_noise": float(s_noise),
    }


def _candidate_key_for_piecewise(stage: str, extra_args: dict[str, Any]) -> str:
    return (
        f"{stage}_piecewise_smax{_format_token(extra_args['sigma_max'])}"
        f"_t{_format_token(extra_args['sigma_transition'])}"
        f"_h{int(extra_args['num_steps_high'])}"
        f"_l{int(extra_args['num_steps_low'])}"
    )


def _candidate_from_extra_args(
    *,
    stage: str,
    profile: CheckpointProfile,
    candidate_key: str,
    candidate_label: str,
    family: str,
    run_date: str,
    extra_args: dict[str, Any],
    study: str = "o96_o320_piecewise_scheduler_search",
) -> SchedulerCandidate:
    sigma_schedule = materialize_sigma_schedule(extra_args)
    return SchedulerCandidate(
        study=study,
        stage=stage,
        checkpoint_id=profile.checkpoint_id,
        checkpoint_short=profile.checkpoint_short,
        checkpoint_path=profile.checkpoint_path,
        stack_flavor=profile.stack_flavor,
        lane=profile.lane,
        baseline_slug=profile.baseline_slug,
        candidate_key=candidate_key,
        candidate_label=candidate_label,
        family=family,
        run_id=_build_run_id(profile, run_date, candidate_key),
        total_steps=int(extra_args["num_steps"]),
        extra_args=dict(extra_args),
        sigma_schedule=sigma_schedule,
    )


def build_control_candidates(
    profiles: Sequence[CheckpointProfile],
    *,
    run_date: str | None = None,
) -> list[SchedulerCandidate]:
    run_date = run_date or _default_run_date()
    candidates: list[SchedulerCandidate] = []
    for profile in profiles:
        for control in CONTROL_CANDIDATES:
            candidates.append(
                _candidate_from_extra_args(
                    stage="control",
                    profile=profile,
                    candidate_key=str(control["candidate_key"]),
                    candidate_label=str(control["label"]),
                    family="control",
                    run_date=run_date,
                    extra_args=dict(control["extra_args"]),
                )
            )
    return candidates


def build_stage0_manifest(
    profiles: Sequence[CheckpointProfile],
    *,
    run_date: str | None = None,
    sigma_transition: float = 10.0,
    sigma_max_values: Sequence[float] = DEFAULT_STAGE0_SIGMA_MAX_VALUES,
    num_steps_high_values: Sequence[int] = DEFAULT_STAGE0_NUM_STEPS_HIGH_VALUES,
    num_steps_low: int = 20,
    include_controls: bool = True,
) -> dict[str, Any]:
    run_date = run_date or _default_run_date()
    candidates: list[SchedulerCandidate] = []
    for profile in profiles:
        for sigma_max in sigma_max_values:
            for num_steps_high in num_steps_high_values:
                extra_args = _make_piecewise_extra_args(
                    sigma_max=float(sigma_max),
                    sigma_transition=float(sigma_transition),
                    num_steps_high=int(num_steps_high),
                    num_steps_low=int(num_steps_low),
                    stage_tag="stage0",
                )
                candidate_key = _candidate_key_for_piecewise("stage0", extra_args)
                candidates.append(
                    _candidate_from_extra_args(
                        stage="stage0",
                        profile=profile,
                        candidate_key=candidate_key,
                        candidate_label=candidate_key,
                        family="search",
                        run_date=run_date,
                        extra_args=extra_args,
                    )
                )
    if include_controls:
        candidates.extend(build_control_candidates(profiles, run_date=run_date))
    return _manifest_payload("stage0", profiles, candidates, run_date)


def build_stage1_manifest(
    profiles: Sequence[CheckpointProfile],
    best_candidate: SchedulerCandidate | dict[str, Any],
    *,
    run_date: str | None = None,
    transition_values: Sequence[float] = DEFAULT_TRANSITION_CONTROLS,
    ensure_transition_controls: bool = True,
    num_steps_high_values: Sequence[int] | None = None,
    num_steps_low_values: Sequence[int] = DEFAULT_STAGE1_NUM_STEPS_LOW_VALUES,
    include_controls: bool = True,
) -> dict[str, Any]:
    run_date = run_date or _default_run_date()
    best = best_candidate if isinstance(best_candidate, SchedulerCandidate) else SchedulerCandidate.from_dict(best_candidate)
    base_args = dict(best.extra_args)
    sigma_max = float(base_args["sigma_max"])
    num_steps_high_candidates = (
        tuple(int(value) for value in num_steps_high_values)
        if num_steps_high_values is not None
        else (int(base_args["num_steps_high"]),)
    )
    transition_values = (
        _ensure_transition_controls(transition_values)
        if ensure_transition_controls
        else tuple(float(value) for value in transition_values)
    )
    candidates: list[SchedulerCandidate] = []
    for profile in profiles:
        for sigma_transition in transition_values:
            for num_steps_high in num_steps_high_candidates:
                for num_steps_low in num_steps_low_values:
                    extra_args = _make_piecewise_extra_args(
                        sigma_max=sigma_max,
                        sigma_transition=float(sigma_transition),
                        num_steps_high=int(num_steps_high),
                        num_steps_low=int(num_steps_low),
                        sigma_min=float(base_args.get("sigma_min", DEFAULT_SIGMA_MIN)),
                        rho=float(base_args.get("rho", DEFAULT_RHO)),
                        sampler=str(base_args.get("sampler", DEFAULT_SAMPLER)),
                        s_churn=float(base_args.get("S_churn", DEFAULT_S_CHURN)),
                        s_min=float(base_args.get("S_min", DEFAULT_S_MIN)),
                        s_noise=float(base_args.get("S_noise", DEFAULT_S_NOISE)),
                        stage_tag="stage1",
                    )
                    candidate_key = _candidate_key_for_piecewise("stage1", extra_args)
                    candidates.append(
                        _candidate_from_extra_args(
                            stage="stage1",
                            profile=profile,
                            candidate_key=candidate_key,
                            candidate_label=candidate_key,
                            family="search",
                            run_date=run_date,
                            extra_args=extra_args,
                        )
                    )
    if include_controls:
        candidates.extend(build_control_candidates(profiles, run_date=run_date))
    return _manifest_payload("stage1", profiles, candidates, run_date)


def build_stage2_manifest(
    profiles: Sequence[CheckpointProfile],
    best_candidate: SchedulerCandidate | dict[str, Any],
    *,
    run_date: str | None = None,
    sigma_max_values: Sequence[float] = DEFAULT_STAGE0_SIGMA_MAX_VALUES,
    num_steps_high_values: Sequence[int] = DEFAULT_STAGE0_NUM_STEPS_HIGH_VALUES,
    include_controls: bool = True,
) -> dict[str, Any]:
    run_date = run_date or _default_run_date()
    best = best_candidate if isinstance(best_candidate, SchedulerCandidate) else SchedulerCandidate.from_dict(best_candidate)
    base_args = dict(best.extra_args)
    sigma_candidates = _local_neighborhood(sigma_max_values, float(base_args["sigma_max"]))
    high_step_candidates = _local_neighborhood(num_steps_high_values, int(base_args["num_steps_high"]))
    candidates: list[SchedulerCandidate] = []
    for profile in profiles:
        for sigma_max in sigma_candidates:
            for num_steps_high in high_step_candidates:
                extra_args = _make_piecewise_extra_args(
                    sigma_max=float(sigma_max),
                    sigma_transition=float(base_args["sigma_transition"]),
                    num_steps_high=int(num_steps_high),
                    num_steps_low=int(base_args["num_steps_low"]),
                    sigma_min=float(base_args.get("sigma_min", DEFAULT_SIGMA_MIN)),
                    rho=float(base_args.get("rho", DEFAULT_RHO)),
                    sampler=str(base_args.get("sampler", DEFAULT_SAMPLER)),
                    s_churn=float(base_args.get("S_churn", DEFAULT_S_CHURN)),
                    s_min=float(base_args.get("S_min", DEFAULT_S_MIN)),
                    s_noise=float(base_args.get("S_noise", DEFAULT_S_NOISE)),
                    stage_tag="stage2",
                )
                candidate_key = _candidate_key_for_piecewise("stage2", extra_args)
                candidates.append(
                    _candidate_from_extra_args(
                        stage="stage2",
                        profile=profile,
                        candidate_key=candidate_key,
                        candidate_label=candidate_key,
                        family="search",
                        run_date=run_date,
                        extra_args=extra_args,
                    )
                )
    if include_controls:
        candidates.extend(build_control_candidates(profiles, run_date=run_date))
    return _manifest_payload("stage2", profiles, candidates, run_date)


def _local_neighborhood(values: Sequence[float | int], focus: float | int) -> tuple[float | int, ...]:
    ordered = list(values)
    if focus not in ordered:
        ordered.append(focus)
        ordered.sort()
    index = ordered.index(focus)
    left = max(index - 1, 0)
    right = min(index + 2, len(ordered))
    return tuple(ordered[left:right])


def _ensure_transition_controls(values: Sequence[float]) -> tuple[float, ...]:
    merged = {float(value) for value in values}
    merged.update(DEFAULT_TRANSITION_CONTROLS)
    return tuple(sorted(merged))


def _manifest_payload(
    stage: str,
    profiles: Sequence[CheckpointProfile],
    candidates: Sequence[SchedulerCandidate],
    run_date: str,
) -> dict[str, Any]:
    return {
        "study": "o96_o320_piecewise_scheduler_search",
        "stage": stage,
        "lane": DEFAULT_LANE,
        "run_date": run_date,
        "checkpoints": [profile.to_dict() for profile in profiles],
        "candidate_count": len(candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def build_schedule_report_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_candidate in manifest["candidates"]:
        candidate = SchedulerCandidate.from_dict(raw_candidate)
        schedule = candidate.sigma_schedule
        preview = ", ".join(_format_sigma(value) for value in schedule[:5])
        rows.append(
            {
                "stage": candidate.stage,
                "checkpoint_short": candidate.checkpoint_short,
                "candidate_key": candidate.candidate_key,
                "family": candidate.family,
                "total_steps": candidate.total_steps,
                "schedule_type": candidate.extra_args["schedule_type"],
                "sigma_max": candidate.extra_args["sigma_max"],
                "sigma_transition": candidate.extra_args.get("sigma_transition", ""),
                "num_steps_high": candidate.extra_args.get("num_steps_high", ""),
                "num_steps_low": candidate.extra_args.get("num_steps_low", ""),
                "sigma_preview": preview,
            }
        )
    return rows


def write_schedule_report(manifest: dict[str, Any], out_path: str | Path) -> tuple[Path, Path]:
    out_path = Path(out_path)
    csv_path = out_path.with_name(f"{out_path.stem}_schedule.csv")
    md_path = out_path.with_name(f"{out_path.stem}_schedule.md")
    rows = build_schedule_report_rows(manifest)
    fieldnames = list(rows[0].keys()) if rows else [
        "stage",
        "checkpoint_short",
        "candidate_key",
        "family",
        "total_steps",
        "schedule_type",
        "sigma_max",
        "sigma_transition",
        "num_steps_high",
        "num_steps_low",
        "sigma_preview",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {manifest['stage']} schedule report",
        "",
        f"- candidate_count: `{manifest['candidate_count']}`",
        f"- run_date: `{manifest['run_date']}`",
        "",
        "| checkpoint | candidate | family | steps | schedule | sigma_max | transition | preview |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {checkpoint_short} | {candidate_key} | {family} | {total_steps} | {schedule_type} | {sigma_max} | {sigma_transition} | {sigma_preview} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _format_sigma(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def analyze_transition_prior(
    *,
    ell: Sequence[float],
    low_res_power: Sequence[float],
    high_res_power: Sequence[float],
    residual_power: Sequence[float] | None = None,
    fixed_transitions: Sequence[float] = DEFAULT_TRANSITION_CONTROLS,
) -> dict[str, Any]:
    ell_arr = np.asarray(ell, dtype=np.float64)
    low_arr = np.maximum(np.asarray(low_res_power, dtype=np.float64), 1e-12)
    high_arr = np.maximum(np.asarray(high_res_power, dtype=np.float64), 1e-12)
    if ell_arr.ndim != 1 or low_arr.shape != ell_arr.shape or high_arr.shape != ell_arr.shape:
        raise ValueError("ell, low_res_power, and high_res_power must be one-dimensional arrays of the same length.")

    departure_signal = np.abs(np.log(high_arr) - np.log(low_arr))
    if departure_signal.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        smoothed = np.convolve(departure_signal, kernel, mode="same")
    else:
        smoothed = departure_signal

    threshold = max(0.15, float(np.quantile(smoothed, 0.75)))
    if np.any(smoothed >= threshold):
        departure_index = int(np.argmax(smoothed >= threshold))
    else:
        departure_index = int(np.argmax(smoothed))
    departure_ell = float(ell_arr[departure_index])

    residual_focus_ell = departure_ell
    if residual_power is not None:
        residual_arr = np.maximum(np.asarray(residual_power, dtype=np.float64), 0.0)
        if residual_arr.shape != ell_arr.shape:
            raise ValueError("residual_power must have the same shape as ell when provided.")
        residual_focus_ell = float(ell_arr[int(np.argmax(residual_arr))])

    focus_ratio = float(max(departure_ell, residual_focus_ell) / ell_arr.max())
    if focus_ratio < 0.15:
        seed_transition = 100.0
    elif focus_ratio < 0.30:
        seed_transition = 30.0
    elif focus_ratio < 0.55:
        seed_transition = 10.0
    elif focus_ratio < 0.75:
        seed_transition = 3.0
    else:
        seed_transition = 1.0

    transitions = sorted({float(value) for value in fixed_transitions})
    priority_order = sorted(
        transitions,
        key=lambda value: (abs(math.log10(value) - math.log10(seed_transition)), value),
    )
    return {
        "departure_ell": departure_ell,
        "residual_focus_ell": residual_focus_ell,
        "seed_transition": seed_transition,
        "candidate_transitions": transitions,
        "priority_order": priority_order,
        "departure_threshold": threshold,
    }


def evaluate_spectra_guardrail(
    result: dict[str, Any],
    baseline: dict[str, Any],
    *,
    spectra_mean_tolerance: float = 0.003,
    variable_tolerance: float = 0.01,
) -> dict[str, Any]:
    variable_fields = ("10u", "10v", "2t")
    mean_drop = float(baseline["spectra_mean"] - result["spectra_mean"])
    variable_drops = {
        field: float(baseline[f"spectra_{field}"] - result[f"spectra_{field}"])
        for field in variable_fields
    }
    guardrail_pass = mean_drop <= spectra_mean_tolerance and all(
        drop <= variable_tolerance for drop in variable_drops.values()
    )
    return {
        "guardrail_pass": guardrail_pass,
        "spectra_mean_drop": mean_drop,
        "variable_drops": variable_drops,
    }


def rank_scheduler_results(
    payload: dict[str, Any],
    *,
    spectra_mean_tolerance: float = 0.003,
    variable_tolerance: float = 0.01,
) -> list[dict[str, Any]]:
    baseline_metrics = payload["baseline_metrics"]
    results = payload["results"]
    grouped: dict[str, dict[str, Any]] = {}
    for raw in results:
        row = dict(raw)
        checkpoint_short = str(row["checkpoint_short"])
        baseline = dict(baseline_metrics[checkpoint_short])
        guardrail = evaluate_spectra_guardrail(
            row,
            baseline,
            spectra_mean_tolerance=spectra_mean_tolerance,
            variable_tolerance=variable_tolerance,
        )
        enriched = {
            **row,
            **guardrail,
            "idalia_tc_delta": float(row["idalia_tc"] - baseline["idalia_tc"]),
        }
        group = grouped.setdefault(
            str(row["candidate_key"]),
            {
                "candidate_key": str(row["candidate_key"]),
                "candidate_label": str(row.get("candidate_label", row["candidate_key"])),
                "total_steps": int(row["total_steps"]),
                "results_by_checkpoint": [],
            },
        )
        group["results_by_checkpoint"].append(enriched)

    summaries: list[dict[str, Any]] = []
    for summary in grouped.values():
        rows = summary["results_by_checkpoint"]
        guardrail_pass = all(row["guardrail_pass"] for row in rows)
        mean_idalia_delta = float(np.mean([row["idalia_tc_delta"] for row in rows]))
        worst_idalia_delta = float(np.min([row["idalia_tc_delta"] for row in rows]))
        summary["guardrail_pass"] = guardrail_pass
        summary["mean_idalia_tc_delta"] = mean_idalia_delta
        summary["worst_idalia_tc_delta"] = worst_idalia_delta
        summary["failed_checkpoints"] = [
            row["checkpoint_short"] for row in rows if not row["guardrail_pass"]
        ]
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            0 if row["guardrail_pass"] else 1,
            -row["mean_idalia_tc_delta"],
            -row["worst_idalia_tc_delta"],
            row["total_steps"],
            row["candidate_key"],
        )
    )
    return summaries


def write_json_report(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_launch_ledger(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _manifest_candidate_index(manifest: dict[str, Any]) -> dict[tuple[str, str], SchedulerCandidate]:
    index: dict[tuple[str, str], SchedulerCandidate] = {}
    for raw_candidate in manifest["candidates"]:
        candidate = SchedulerCandidate.from_dict(raw_candidate)
        index[(candidate.checkpoint_short, candidate.candidate_key)] = candidate
    return index


def _tc_event_summary(event_payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(event_payload.get("aggregate", {}))
    return {
        "status": str(event_payload.get("status", "")),
        "max_deviation": float(event_payload["max_deviation"]) if "max_deviation" in event_payload else None,
        "mslp_min": float(aggregate["mslp_min"]["proxy_extreme"]) if "mslp_min" in aggregate else None,
        "wind_max": float(aggregate["wind_max"]["proxy_extreme"]) if "wind_max" in aggregate else None,
        "mslp_980_990_count": int(aggregate["mslp_980_990_count"]["proxy"]) if "mslp_980_990_count" in aggregate else None,
        "wind_gt_25_count": int(aggregate["wind_gt_25_count"]["proxy"]) if "wind_gt_25_count" in aggregate else None,
    }


def build_ecmwf_stage_scoreboard(
    manifest: dict[str, Any],
    launch_ledger: Sequence[dict[str, str]],
    *,
    control_candidate_key: str = "control_piecewise30_reference",
    spectra_dirname: str = "spectra_ecmwf",
    tc_filename: str = "proxy_tc_compare.json",
    events: Sequence[str] = ("idalia", "franklin"),
) -> dict[str, Any]:
    candidate_index = _manifest_candidate_index(manifest)
    rows_by_checkpoint: dict[str, list[dict[str, str]]] = {}
    for row in launch_ledger:
        rows_by_checkpoint.setdefault(str(row["checkpoint_short"]), []).append(dict(row))

    scoreboard_rows: list[dict[str, Any]] = []
    for checkpoint_short, checkpoint_rows in sorted(rows_by_checkpoint.items()):
        control_row = next(
            row for row in checkpoint_rows if str(row["candidate_key"]) == control_candidate_key
        )
        control_root = Path(control_row["run_root"])
        control_spectra_root = control_root / spectra_dirname
        control_tc_payload = json.loads((control_root / tc_filename).read_text(encoding="utf-8"))
        control_events = {
            event_name: _tc_event_summary(dict(control_tc_payload.get("events", {}).get(event_name, {})))
            for event_name in events
        }

        for ledger_row in checkpoint_rows:
            run_root = Path(ledger_row["run_root"])
            candidate = candidate_index[(checkpoint_short, str(ledger_row["candidate_key"]))]
            spectra = _load_ecmwf_spectra_metrics(
                run_root / spectra_dirname,
                reference_root=control_spectra_root,
            )
            tc_payload = json.loads((run_root / tc_filename).read_text(encoding="utf-8"))

            row: dict[str, Any] = {
                "checkpoint_short": checkpoint_short,
                "candidate_key": candidate.candidate_key,
                "candidate_label": candidate.candidate_label,
                "family": candidate.family,
                "run_id": candidate.run_id,
                "total_steps": candidate.total_steps,
                "schedule_type": str(candidate.extra_args["schedule_type"]),
                "sigma_max": float(candidate.extra_args["sigma_max"]),
                "sigma_transition": float(candidate.extra_args.get("sigma_transition"))
                if "sigma_transition" in candidate.extra_args
                else None,
                "num_steps_high": int(candidate.extra_args.get("num_steps_high"))
                if "num_steps_high" in candidate.extra_args
                else None,
                "num_steps_low": int(candidate.extra_args.get("num_steps_low"))
                if "num_steps_low" in candidate.extra_args
                else None,
                "spectra_vs_control_mean": spectra.get("mean"),
                "spectra_vs_control_10u": spectra.get("10u"),
                "spectra_vs_control_10v": spectra.get("10v"),
                "spectra_vs_control_2t": spectra.get("2t"),
                "spectra_vs_control_coverage": spectra.get("coverage"),
                "spectra_vs_control_n_curves": spectra.get("n_curves"),
                "spectra_reference_run_id": str(control_row["run_id"]),
                "run_root": str(run_root),
            }

            tc_events = {
                event_name: _tc_event_summary(dict(tc_payload.get("events", {}).get(event_name, {})))
                for event_name in events
            }
            for event_name in events:
                event_metrics = tc_events[event_name]
                control_metrics = control_events[event_name]
                prefix = event_name
                row[f"{prefix}_status"] = event_metrics["status"]
                row[f"{prefix}_max_deviation"] = event_metrics["max_deviation"]
                row[f"{prefix}_mslp_min"] = event_metrics["mslp_min"]
                row[f"{prefix}_wind_max"] = event_metrics["wind_max"]
                row[f"{prefix}_mslp_980_990_count"] = event_metrics["mslp_980_990_count"]
                row[f"{prefix}_wind_gt_25_count"] = event_metrics["wind_gt_25_count"]

                for metric_key in ("mslp_min", "wind_max", "mslp_980_990_count", "wind_gt_25_count"):
                    metric_value = event_metrics[metric_key]
                    control_value = control_metrics[metric_key]
                    delta_key = f"{prefix}_{metric_key}_delta_vs_control"
                    row[delta_key] = (
                        None
                        if metric_value is None or control_value is None
                        else metric_value - control_value
                    )
            scoreboard_rows.append(row)

    scoreboard_rows.sort(
        key=lambda row: (
            row["checkpoint_short"],
            0 if row["candidate_key"] == control_candidate_key else 1,
            999.0 if row["spectra_vs_control_mean"] is None else float(row["spectra_vs_control_mean"]),
            int(row["total_steps"]),
            row["candidate_key"],
        )
    )
    return {
        "study": manifest.get("study", "o96_o320_piecewise_scheduler_search"),
        "stage": manifest.get("stage", ""),
        "scoreboard_kind": "ecmwf_spectra_vs_control",
        "control_candidate_key": control_candidate_key,
        "events": list(events),
        "row_count": len(scoreboard_rows),
        "rows": scoreboard_rows,
    }


def write_ecmwf_stage_scoreboard(
    payload: dict[str, Any],
    out_path: str | Path,
) -> tuple[Path, Path, Path]:
    out_path = Path(out_path)
    json_path = write_json_report(payload, out_path)
    csv_path = out_path.with_suffix(".csv")
    md_path = out_path.with_suffix(".md")

    fieldnames = [
        "checkpoint_short",
        "candidate_key",
        "family",
        "total_steps",
        "schedule_type",
        "sigma_max",
        "sigma_transition",
        "num_steps_high",
        "num_steps_low",
        "spectra_vs_control_mean",
        "spectra_vs_control_10u",
        "spectra_vs_control_10v",
        "spectra_vs_control_2t",
        "spectra_vs_control_coverage",
        "idalia_wind_max_delta_vs_control",
        "idalia_mslp_min_delta_vs_control",
        "idalia_wind_gt_25_count_delta_vs_control",
        "idalia_mslp_980_990_count_delta_vs_control",
        "franklin_wind_max_delta_vs_control",
        "franklin_mslp_min_delta_vs_control",
        "franklin_wind_gt_25_count_delta_vs_control",
        "franklin_mslp_980_990_count_delta_vs_control",
        "idalia_max_deviation",
        "franklin_max_deviation",
        "run_id",
        "run_root",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["rows"])

    def _format_cell(value: Any, *, places: int = 5) -> str:
        if value is None:
            return "na"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{places}f}"
        return str(value)

    lines = [
        f"# {payload['stage']} ECMWF scheduler scoreboard",
        "",
        f"- study: `{payload['study']}`",
        f"- control_candidate: `{payload['control_candidate_key']}`",
        "- spectra columns are direct ECMWF mean-curve relative-L2 against the checkpoint-matched control run; lower is better",
        "- TC delta columns are candidate minus control; lower `mslp_min` is deeper, higher `wind_max` / counts is stronger",
        "",
    ]
    checkpoints = sorted({str(row["checkpoint_short"]) for row in payload["rows"]})
    for checkpoint_short in checkpoints:
        lines.extend(
            [
                f"## {checkpoint_short}",
                "",
                "| candidate | family | steps | smax | t | h | l | spectra mean | 10u | 10v | 2t | I wind d | I mslp d | I windct d | I mslpct d | F wind d | F mslp d | F windct d | F mslpct d |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in payload["rows"]:
            if str(row["checkpoint_short"]) != checkpoint_short:
                continue
            lines.append(
                "| {candidate_key} | {family} | {total_steps} | {sigma_max} | {sigma_transition} | {num_steps_high} | {num_steps_low} | {spectra_mean} | {spectra_10u} | {spectra_10v} | {spectra_2t} | {idalia_wind} | {idalia_mslp} | {idalia_wind_ct} | {idalia_mslp_ct} | {franklin_wind} | {franklin_mslp} | {franklin_wind_ct} | {franklin_mslp_ct} |".format(
                    candidate_key=row["candidate_key"],
                    family=row["family"],
                    total_steps=row["total_steps"],
                    sigma_max=_format_cell(row["sigma_max"], places=0),
                    sigma_transition=_format_cell(row["sigma_transition"], places=0),
                    num_steps_high=_format_cell(row["num_steps_high"], places=0),
                    num_steps_low=_format_cell(row["num_steps_low"], places=0),
                    spectra_mean=_format_cell(row["spectra_vs_control_mean"]),
                    spectra_10u=_format_cell(row["spectra_vs_control_10u"]),
                    spectra_10v=_format_cell(row["spectra_vs_control_10v"]),
                    spectra_2t=_format_cell(row["spectra_vs_control_2t"]),
                    idalia_wind=_format_cell(row["idalia_wind_max_delta_vs_control"], places=2),
                    idalia_mslp=_format_cell(row["idalia_mslp_min_delta_vs_control"], places=2),
                    idalia_wind_ct=_format_cell(row["idalia_wind_gt_25_count_delta_vs_control"], places=0),
                    idalia_mslp_ct=_format_cell(row["idalia_mslp_980_990_count_delta_vs_control"], places=0),
                    franklin_wind=_format_cell(row["franklin_wind_max_delta_vs_control"], places=2),
                    franklin_mslp=_format_cell(row["franklin_mslp_min_delta_vs_control"], places=2),
                    franklin_wind_ct=_format_cell(row["franklin_wind_gt_25_count_delta_vs_control"], places=0),
                    franklin_mslp_ct=_format_cell(row["franklin_mslp_980_990_count_delta_vs_control"], places=0),
                )
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheduler-study helpers for o96->o320 piecewise search.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="Build a stage manifest for the scheduler study.")
    build.add_argument("--stage", choices=["stage0", "stage1", "stage2"], required=True)
    build.add_argument("--checkpoint-profiles", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--run-date", default="")
    build.add_argument("--sigma-transition", type=float, default=10.0)
    build.add_argument("--sigma-max-values", default="1000,10000,100000")
    build.add_argument("--num-steps-high-values", default="3,5,7")
    build.add_argument("--stage1-num-steps-high-values", default="")
    build.add_argument("--num-steps-low", type=int, default=20)
    build.add_argument("--num-steps-low-values", default="12,16,20")
    build.add_argument("--transition-values", default="1,3,10,30,100")
    build.add_argument("--exact-transition-values", action="store_true")
    build.add_argument("--best-candidate-json", default="")
    build.add_argument("--exclude-controls", action="store_true")

    prior = sub.add_parser("analyze-transition-prior", help="Compute a spectra-informed transition prior.")
    prior.add_argument("--spectra-json", required=True)
    prior.add_argument("--out", required=True)

    rank = sub.add_parser("rank-results", help="Rank scheduler-study candidates from merged metrics.")
    rank.add_argument("--results-json", required=True)
    rank.add_argument("--out", required=True)
    rank.add_argument("--spectra-mean-tolerance", type=float, default=0.003)
    rank.add_argument("--variable-tolerance", type=float, default=0.01)

    scoreboard = sub.add_parser(
        "build-ecmwf-scoreboard",
        help="Build an ECMWF-spectra scheduler scoreboard from a manifest plus launch ledger.",
    )
    scoreboard.add_argument("--manifest-json", required=True)
    scoreboard.add_argument("--launch-ledger", required=True)
    scoreboard.add_argument("--out", required=True)
    scoreboard.add_argument("--control-candidate-key", default="control_piecewise30_reference")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-manifest":
        profiles = load_checkpoint_profiles(args.checkpoint_profiles)
        run_date = args.run_date or _default_run_date()
        include_controls = not args.exclude_controls
        if args.stage == "stage0":
            manifest = build_stage0_manifest(
                profiles,
                run_date=run_date,
                sigma_transition=args.sigma_transition,
                sigma_max_values=_parse_float_csv(args.sigma_max_values),
                num_steps_high_values=_parse_int_csv(args.num_steps_high_values),
                num_steps_low=args.num_steps_low,
                include_controls=include_controls,
            )
        else:
            if not args.best_candidate_json:
                parser.error("--best-candidate-json is required for stage1 and stage2.")
            best_candidate = _load_candidate_payload(args.best_candidate_json)
            if args.stage == "stage1":
                manifest = build_stage1_manifest(
                    profiles,
                    best_candidate,
                    run_date=run_date,
                    transition_values=_parse_float_csv(args.transition_values),
                    ensure_transition_controls=not args.exact_transition_values,
                    num_steps_high_values=(
                        _parse_int_csv(args.stage1_num_steps_high_values)
                        if args.stage1_num_steps_high_values
                        else None
                    ),
                    num_steps_low_values=_parse_int_csv(args.num_steps_low_values),
                    include_controls=include_controls,
                )
            else:
                manifest = build_stage2_manifest(
                    profiles,
                    best_candidate,
                    run_date=run_date,
                    sigma_max_values=_parse_float_csv(args.sigma_max_values),
                    num_steps_high_values=_parse_int_csv(args.num_steps_high_values),
                    include_controls=include_controls,
                )
        out_path = write_json_report(manifest, args.out)
        write_schedule_report(manifest, out_path)
        print(out_path)
        return

    if args.command == "analyze-transition-prior":
        raw = json.loads(Path(args.spectra_json).read_text(encoding="utf-8"))
        result = analyze_transition_prior(
            ell=raw["ell"],
            low_res_power=raw["low_res_power"],
            high_res_power=raw["high_res_power"],
            residual_power=raw.get("residual_power"),
        )
        out_path = write_json_report(result, args.out)
        print(out_path)
        return

    if args.command == "rank-results":
        payload = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
        ranked = rank_scheduler_results(
            payload,
            spectra_mean_tolerance=args.spectra_mean_tolerance,
            variable_tolerance=args.variable_tolerance,
        )
        out_path = write_json_report({"ranked_candidates": ranked}, args.out)
        print(out_path)
        return

    if args.command == "build-ecmwf-scoreboard":
        manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
        launch_ledger = load_launch_ledger(args.launch_ledger)
        payload = build_ecmwf_stage_scoreboard(
            manifest,
            launch_ledger,
            control_candidate_key=args.control_candidate_key,
        )
        json_path, csv_path, md_path = write_ecmwf_stage_scoreboard(payload, args.out)
        print(json_path)
        print(csv_path)
        print(md_path)
        return


if __name__ == "__main__":
    main()
