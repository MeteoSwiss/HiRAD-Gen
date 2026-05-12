from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch_geometric.utils import scatter

from eval.archive.jobs.generate_predictions_25_files import BundleKey
from eval.archive.jobs.generate_predictions_25_files import discover_bundles
from eval._backends.region_plotting.local_plotting import get_region_ds
from eval._backends.region_plotting.plot_regions import PREDICTION_REGION_BOXES
from manual_inference.input_data_construction.bundle import extract_target_from_bundle
from manual_inference.input_data_construction.bundle import load_inputs_from_bundle_numpy
from manual_inference.prediction.predict import _load_objects
from manual_inference.prediction.predict import _validate_bundle_hres_contract

DEFAULT_EXTREME_COUNT = 2
DEFAULT_MODERATE_COUNT = 1
DEFAULT_CONTROL_COUNT = 1
DEFAULT_SEED = 0
DEFAULT_CONTROL_REGION = "amazon_forest"
DEFAULT_CONTROL_DATE = "20230826"
DEFAULT_CONTROL_STEP = 120
DEFAULT_CONTROL_MEMBER = 1
PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")

ACTIVE_CONTEXT: "RunRecorder | None" = None


@dataclass(frozen=True)
class SelectedCase:
    case_id: str
    category: str
    date: str
    step: int
    member: int
    bundle_path: str
    primary_region: str
    truth_max_wind10m_ms: float | None
    truth_min_mslp_pa: float | None
    lowdec_max_wind10m_ms: float | None
    lowdec_min_mslp_pa: float | None
    highdec_max_wind10m_ms: float | None
    highdec_min_mslp_pa: float | None
    selection_reason: str


@dataclass
class PredictionArrays:
    y_pred: np.ndarray
    y_true: np.ndarray | None
    weather_states: list[str]
    x_lres_raw: np.ndarray
    lon_lres: np.ndarray
    lat_lres: np.ndarray
    lon_hres: np.ndarray
    lat_hres: np.ndarray


@dataclass
class AttentionAccumulator:
    entropy_sum: float = 0.0
    max_weight_sum: float = 0.0
    count: int = 0

    def update(self, alpha: torch.Tensor, index: torch.Tensor, size_i: int | None) -> None:
        alpha = alpha.detach()
        index = index.detach()
        dim_size = int(size_i) if size_i is not None else None
        alpha_safe = alpha.clamp_min(1.0e-12)
        entropy_terms = -(alpha_safe * alpha_safe.log())
        entropies = scatter(entropy_terms, index, dim=0, dim_size=dim_size, reduce="sum")
        max_weights = scatter(alpha, index, dim=0, dim_size=dim_size, reduce="max")
        self.entropy_sum += float(entropies.sum().item())
        self.max_weight_sum += float(max_weights.sum().item())
        self.count += int(entropies.numel())

    def as_metrics(self) -> dict[str, float]:
        if self.count <= 0:
            return {
                "attention_entropy": math.nan,
                "attention_max_weight": math.nan,
            }
        return {
            "attention_entropy": self.entropy_sum / self.count,
            "attention_max_weight": self.max_weight_sum / self.count,
        }


@dataclass
class RunRecorder:
    case: SelectedCase
    checkpoint_label: str
    block_rows: list[dict[str, Any]] = field(default_factory=list)
    attention_stats: dict[str, AttentionAccumulator] = field(default_factory=dict)

    def record_block(
        self,
        *,
        module_name: str,
        phase: str,
        block_index: int,
        depth_order: int,
        block_output: torch.Tensor,
        attention_delta: torch.Tensor,
        attention_input: torch.Tensor,
        mlp_delta: torch.Tensor,
        mlp_input: torch.Tensor,
    ) -> None:
        self.block_rows.append(
            {
                "case_id": self.case.case_id,
                "category": self.case.category,
                "checkpoint": self.checkpoint_label,
                "phase": phase,
                "block_index": int(block_index),
                "depth_order": int(depth_order),
                "module_name": module_name,
                "block_output_rms": tensor_rms(block_output),
                "block_output_max_abs": tensor_max_abs(block_output),
                "attention_delta_ratio": tensor_ratio(attention_delta, attention_input),
                "mlp_delta_ratio": tensor_ratio(mlp_delta, mlp_input),
            }
        )

    def record_attention(
        self,
        *,
        module_name: str,
        alpha: torch.Tensor,
        index: torch.Tensor,
        size_i: int | None,
    ) -> None:
        acc = self.attention_stats.setdefault(module_name, AttentionAccumulator())
        acc.update(alpha=alpha, index=index, size_i=size_i)

    def finalize_rows(self) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for row in self.block_rows:
            attn_metrics = self.attention_stats.get(row["module_name"])
            if attn_metrics is None:
                row["attention_entropy"] = math.nan
                row["attention_max_weight"] = math.nan
            else:
                row.update(attn_metrics.as_metrics())
            finalized.append(row)
        return finalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal mechanistic comparison between two o96->o320 checkpoints "
            "using matched cases, lightweight transformer instrumentation, and finite differences."
        )
    )
    parser.add_argument("--name-ckpt-a", required=True)
    parser.add_argument("--name-ckpt-b", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--predictions-a", required=True)
    parser.add_argument("--predictions-b", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--event", default="idalia")
    parser.add_argument("--control-region", default=DEFAULT_CONTROL_REGION)
    parser.add_argument("--extreme-count", type=int, default=DEFAULT_EXTREME_COUNT)
    parser.add_argument("--moderate-count", type=int, default=DEFAULT_MODERATE_COUNT)
    parser.add_argument("--control-count", type=int, default=DEFAULT_CONTROL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--validation-frequency", default="50h")
    parser.add_argument("--sampler-json", required=True)
    parser.add_argument("--case-manifest-in", default="")
    parser.add_argument("--skip-case-selection", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    return parser


def tensor_rms(t: torch.Tensor) -> float:
    arr = t.detach()
    return float(torch.sqrt(torch.mean(arr * arr)).item())


def tensor_max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def tensor_ratio(delta: torch.Tensor, baseline: torch.Tensor, eps: float = 1.0e-12) -> float:
    delta_norm = float(torch.linalg.vector_norm(delta.detach()).item())
    base_norm = float(torch.linalg.vector_norm(baseline.detach()).item())
    return delta_norm / max(base_norm, eps)


def _case_key(date: str, step: int, member: int) -> str:
    return f"{date}_step{int(step):03d}_mem{int(member):02d}"


def _parse_prediction_name(path: Path) -> tuple[str, int] | None:
    match = PRED_RE.match(path.name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _slice_sample(ds: xr.Dataset) -> xr.Dataset:
    if "sample" in ds.dims:
        return ds.isel(sample=0)
    return ds


def _member_values(ds: xr.Dataset) -> list[int]:
    if "ensemble_member" not in ds.dims:
        return [1]
    values = np.asarray(ds["ensemble_member"].values)
    out: list[int] = []
    for idx, raw in enumerate(values):
        if isinstance(raw, (np.integer, int)):
            val = int(raw)
            if 1 <= val <= 99:
                out.append(val)
            elif 0 <= val <= 98:
                out.append(val + 1)
            else:
                out.append(idx + 1)
        else:
            text = str(raw)
            digits = "".join(ch for ch in text if ch.isdigit())
            out.append(int(digits) if digits else idx + 1)
    return out


def _member_array(ds: xr.Dataset, variable: str, member_number: int, weather_state: str) -> np.ndarray:
    da = ds[variable].sel(weather_state=weather_state)
    if "ensemble_member" in da.dims:
        return np.asarray(da.sel(ensemble_member=member_number).values, dtype=np.float32)
    return np.asarray(da.values, dtype=np.float32)


def _region_metrics_from_prediction(ds_region: xr.Dataset, variable: str, member_number: int) -> tuple[float, float]:
    u = _member_array(ds_region, variable, member_number, "10u")
    v = _member_array(ds_region, variable, member_number, "10v")
    msl = _member_array(ds_region, variable, member_number, "msl")
    wind = np.sqrt(u**2 + v**2)
    return float(np.nanmax(wind)), float(np.nanmin(msl))


def _storm_region_name(event_name: str) -> str:
    centered = f"{event_name}_center"
    if centered in PREDICTION_REGION_BOXES:
        return centered
    return event_name


def _build_case_table(
    *,
    predictions_a: Path,
    predictions_b: Path,
    event_name: str,
) -> pd.DataFrame:
    storm_region = _storm_region_name(event_name)
    rows: list[dict[str, Any]] = []
    for pred_a_path in sorted(predictions_a.glob("predictions_*.nc")):
        parsed = _parse_prediction_name(pred_a_path)
        if parsed is None:
            continue
        date, step = parsed
        pred_b_path = predictions_b / pred_a_path.name
        if not pred_b_path.exists():
            continue
        with xr.open_dataset(pred_a_path) as ds_a_raw, xr.open_dataset(pred_b_path) as ds_b_raw:
            ds_a = _slice_sample(ds_a_raw)
            ds_b = _slice_sample(ds_b_raw)
            ds_a_region = get_region_ds(ds_a, storm_region)
            ds_b_region = get_region_ds(ds_b, storm_region)
            members = _member_values(ds_a_region)
            for member_number in members:
                truth_max_wind, truth_min_mslp = _region_metrics_from_prediction(ds_a_region, "y", member_number)
                low_max_wind, low_min_mslp = _region_metrics_from_prediction(ds_a_region, "y_pred", member_number)
                high_max_wind, high_min_mslp = _region_metrics_from_prediction(ds_b_region, "y_pred", member_number)
                rows.append(
                    {
                        "date": date,
                        "step": int(step),
                        "member": int(member_number),
                        "truth_max_wind10m_ms": truth_max_wind,
                        "truth_min_mslp_pa": truth_min_mslp,
                        "lowdec_max_wind10m_ms": low_max_wind,
                        "lowdec_min_mslp_pa": low_min_mslp,
                        "highdec_max_wind10m_ms": high_max_wind,
                        "highdec_min_mslp_pa": high_min_mslp,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No paired prediction files found under {predictions_a} and {predictions_b}"
        )
    df = pd.DataFrame(rows)
    df["wind_error_low"] = (df["lowdec_max_wind10m_ms"] - df["truth_max_wind10m_ms"]).abs()
    df["wind_error_high"] = (df["highdec_max_wind10m_ms"] - df["truth_max_wind10m_ms"]).abs()
    df["mslp_error_low_hpa"] = (df["lowdec_min_mslp_pa"] - df["truth_min_mslp_pa"]).abs() / 100.0
    df["mslp_error_high_hpa"] = (df["highdec_min_mslp_pa"] - df["truth_min_mslp_pa"]).abs() / 100.0
    wind_scale = max(
        float(pd.concat([df["wind_error_low"], df["wind_error_high"]]).median()),
        1.0,
    )
    mslp_scale = max(
        float(pd.concat([df["mslp_error_low_hpa"], df["mslp_error_high_hpa"]]).median()),
        1.0,
    )
    df["composite_error_low"] = (df["wind_error_low"] / wind_scale) + (
        df["mslp_error_low_hpa"] / mslp_scale
    )
    df["composite_error_high"] = (df["wind_error_high"] / wind_scale) + (
        df["mslp_error_high_hpa"] / mslp_scale
    )
    wind_std = max(float(df["truth_max_wind10m_ms"].std(ddof=0)), 1.0)
    msl_std = max(float(df["truth_min_mslp_pa"].std(ddof=0)), 100.0)
    df["truth_extremeness"] = (
        (df["truth_max_wind10m_ms"] - float(df["truth_max_wind10m_ms"].mean())) / wind_std
        + (float(df["truth_min_mslp_pa"].mean()) - df["truth_min_mslp_pa"]) / msl_std
    )
    df["lowdec_improvement"] = df["composite_error_high"] - df["composite_error_low"]
    return df.sort_values(["truth_extremeness", "lowdec_improvement"], ascending=[False, False]).reset_index(drop=True)


def _select_extreme_cases(df: pd.DataFrame, count: int) -> list[pd.Series]:
    candidates = df.head(max(10, count * 4)).copy()
    candidates = candidates.sort_values(
        ["lowdec_improvement", "truth_extremeness"],
        ascending=[False, False],
    )
    return [row for _, row in candidates.head(count).iterrows()]


def _select_moderate_cases(df: pd.DataFrame, count: int, exclude: set[str]) -> list[pd.Series]:
    if count <= 0:
        return []
    low_q = float(df["truth_extremeness"].quantile(0.35))
    high_q = float(df["truth_extremeness"].quantile(0.65))
    moderate = df[
        (df["truth_extremeness"] >= low_q)
        & (df["truth_extremeness"] <= high_q)
        & (df["truth_max_wind10m_ms"] >= 15.0)
    ].copy()
    if moderate.empty:
        moderate = df.copy()
    target = float(moderate["truth_extremeness"].median())
    moderate["distance_to_median"] = (moderate["truth_extremeness"] - target).abs()
    moderate = moderate.sort_values(["distance_to_median", "lowdec_improvement"])
    selected: list[pd.Series] = []
    for _, row in moderate.iterrows():
        key = _case_key(row["date"], int(row["step"]), int(row["member"]))
        if key in exclude:
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _select_control_cases(
    *,
    input_root: Path,
    predictions_a: Path,
    predictions_b: Path,
    count: int,
    control_region: str,
) -> list[SelectedCase]:
    if count <= 0:
        return []
    control_cases: list[SelectedCase] = []
    default_files = [
        (DEFAULT_CONTROL_DATE, DEFAULT_CONTROL_STEP, DEFAULT_CONTROL_MEMBER),
    ]
    bundle_map = discover_bundles(input_root, recursive=True)
    for date, step, member in default_files:
        if len(control_cases) >= count:
            break
        pred_name = f"predictions_{date}_step{int(step):03d}.nc"
        pred_a_path = predictions_a / pred_name
        pred_b_path = predictions_b / pred_name
        if not pred_a_path.exists() or not pred_b_path.exists():
            continue
        bundle_path = bundle_map.get(BundleKey(date=date, step=int(step), member=int(member)))
        if bundle_path is None:
            continue
        control_cases.append(
            SelectedCase(
                case_id=f"control_{date}_step{int(step):03d}_mem{int(member):02d}",
                category="control",
                date=date,
                step=int(step),
                member=int(member),
                bundle_path=str(bundle_path),
                primary_region=control_region,
                truth_max_wind10m_ms=None,
                truth_min_mslp_pa=None,
                lowdec_max_wind10m_ms=None,
                lowdec_min_mslp_pa=None,
                highdec_max_wind10m_ms=None,
                highdec_min_mslp_pa=None,
                selection_reason="fixed control slice in amazon_forest for non-storm comparison",
            )
        )
    if len(control_cases) < count:
        raise FileNotFoundError(
            f"Could not resolve {count} control case(s) from {input_root} and prediction roots."
        )
    return control_cases


def _rows_to_cases(
    *,
    rows: list[pd.Series],
    category: str,
    input_root: Path,
    event_name: str,
) -> list[SelectedCase]:
    bundle_map = discover_bundles(input_root, recursive=True)
    cases: list[SelectedCase] = []
    for idx, row in enumerate(rows, start=1):
        date = str(row["date"])
        step = int(row["step"])
        member = int(row["member"])
        bundle_path = bundle_map.get(BundleKey(date=date, step=step, member=member))
        if bundle_path is None:
            raise FileNotFoundError(
                f"Missing bundle for selected case date={date} step={step} member={member}"
            )
        cases.append(
            SelectedCase(
                case_id=f"{category}_{idx}_{date}_step{step:03d}_mem{member:02d}",
                category=category,
                date=date,
                step=step,
                member=member,
                bundle_path=str(bundle_path),
                primary_region=_storm_region_name(event_name),
                truth_max_wind10m_ms=float(row["truth_max_wind10m_ms"]),
                truth_min_mslp_pa=float(row["truth_min_mslp_pa"]),
                lowdec_max_wind10m_ms=float(row["lowdec_max_wind10m_ms"]),
                lowdec_min_mslp_pa=float(row["lowdec_min_mslp_pa"]),
                highdec_max_wind10m_ms=float(row["highdec_max_wind10m_ms"]),
                highdec_min_mslp_pa=float(row["highdec_min_mslp_pa"]),
                selection_reason=(
                    "strong truth extremeness and low-decay improvement"
                    if category == "extreme"
                    else "mid-strength cyclone representative"
                ),
            )
        )
    return cases


def select_cases(
    *,
    input_root: Path,
    predictions_a: Path,
    predictions_b: Path,
    event_name: str,
    control_region: str,
    extreme_count: int,
    moderate_count: int,
    control_count: int,
) -> list[SelectedCase]:
    case_df = _build_case_table(
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        event_name=event_name,
    )
    extreme_rows = _select_extreme_cases(case_df, extreme_count)
    used_keys = {
        _case_key(str(row["date"]), int(row["step"]), int(row["member"]))
        for row in extreme_rows
    }
    moderate_rows = _select_moderate_cases(case_df, moderate_count, used_keys)
    selected = _rows_to_cases(rows=extreme_rows, category="extreme", input_root=input_root, event_name=event_name)
    selected.extend(_rows_to_cases(rows=moderate_rows, category="moderate", input_root=input_root, event_name=event_name))
    selected.extend(
        _select_control_cases(
            input_root=input_root,
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            count=control_count,
            control_region=control_region,
        )
    )
    return selected


def load_case_manifest(path: Path) -> list[SelectedCase]:
    df = pd.read_csv(path)
    return [SelectedCase(**row) for row in df.to_dict(orient="records")]


def save_case_manifest(cases: list[SelectedCase], out_path: Path) -> None:
    df = pd.DataFrame([asdict(case) for case in cases])
    df.to_csv(out_path, index=False)


def _build_phase_meta(module_name: str) -> tuple[str, int, int]:
    if module_name == "model.encoder.proc":
        return "encoder", 0, 0
    if module_name == "model.decoder.proc":
        return "decoder", 0, 17
    match = re.match(r"model\.processor\.proc\.(\d+)\.blocks\.(\d+)$", module_name)
    if match is None:
        raise ValueError(f"Unsupported block module name: {module_name}")
    stack = int(match.group(1))
    block = int(match.group(2))
    depth = stack * 8 + block + 1
    return "processor", block, depth


def annotate_model_modules(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in {"GraphTransformerMapperBlock", "GraphTransformerProcessorBlock"}:
            phase, block_index, depth_order = _build_phase_meta(name)
            setattr(
                module,
                "_mechanistic_diag_meta",
                {
                    "module_name": name,
                    "phase": phase,
                    "block_index": block_index,
                    "depth_order": depth_order,
                },
            )
        elif cls_name == "GraphTransformerConv":
            block_name = name[: -len(".conv")] if name.endswith(".conv") else name
            phase, block_index, depth_order = _build_phase_meta(block_name)
            setattr(
                module,
                "_mechanistic_diag_meta",
                {
                    "module_name": block_name,
                    "phase": phase,
                    "block_index": block_index,
                    "depth_order": depth_order,
                },
            )


@contextlib.contextmanager
def instrumentation_context(model: torch.nn.Module, recorder: RunRecorder):
    from anemoi.models.layers.block import GraphTransformerMapperBlock
    from anemoi.models.layers.block import GraphTransformerProcessorBlock
    from anemoi.models.layers.conv import GraphTransformerConv
    import einops
    from torch.nn.functional import dropout
    from torch_geometric.utils import softmax

    annotate_model_modules(model)

    original_mapper_forward = GraphTransformerMapperBlock.forward
    original_processor_forward = GraphTransformerProcessorBlock.forward
    original_message = GraphTransformerConv.message

    def mapper_forward(
        self,
        x,
        edge_attr,
        edge_index,
        shapes,
        batch_size,
        size,
        model_comm_group=None,
        cond=None,
        **layer_kwargs,
    ):
        x_skip = x
        cond_src_kwargs = {"cond": cond[0]} if cond is not None else {}
        cond_dst_kwargs = {"cond": cond[1]} if cond is not None else {}
        x = (
            self.layer_norm_attention_src(x[0], **cond_src_kwargs),
            self.layer_norm_attention_dest(x[1], **cond_dst_kwargs),
        )
        x_r = self.lin_self(x[1])
        query, key, value, edges = self.get_qkve(x, edge_attr)
        if self.shard_strategy == "heads":
            query, key, value, edges = self.shard_qkve_heads(
                query, key, value, edges, shapes, batch_size, model_comm_group
            )
        else:
            query, key, value, edges = self.prepare_qkve_edge_sharding(
                query, key, value, edges
            )
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        attn_out = self.attention_block(query, key, value, edges, edge_index, size, num_chunks=1)
        if self.shard_strategy == "heads":
            attn_out = self.shard_output_seq(attn_out, shapes, batch_size, model_comm_group)
        else:
            attn_out = einops.rearrange(attn_out, "nodes heads vars -> nodes (heads vars)")
        attention_delta = self.projection(attn_out + x_r)
        out = attention_delta + x_skip[1]
        mlp_delta = self.run_node_dst_mlp(out, **cond_dst_kwargs)
        nodes_new_dst = mlp_delta + out
        if self.update_src_nodes:
            nodes_new_src = self.run_node_src_mlp(x_skip[0], **cond_dst_kwargs) + x_skip[0]
        else:
            nodes_new_src = x_skip[0]
        meta = getattr(self, "_mechanistic_diag_meta", None)
        if ACTIVE_CONTEXT is not None and meta is not None:
            ACTIVE_CONTEXT.record_block(
                module_name=meta["module_name"],
                phase=meta["phase"],
                block_index=meta["block_index"],
                depth_order=meta["depth_order"],
                block_output=nodes_new_dst,
                attention_delta=attention_delta,
                attention_input=x_skip[1],
                mlp_delta=mlp_delta,
                mlp_input=out,
            )
        return (nodes_new_src, nodes_new_dst), edge_attr

    def processor_forward(
        self,
        x,
        edge_attr,
        edge_index,
        shapes,
        batch_size,
        size,
        model_comm_group=None,
        cond=None,
    ):
        x_skip = x
        cond_kwargs = {"cond": cond} if cond is not None else {}
        x = self.layer_norm_attention(x, **cond_kwargs)
        x_r = self.lin_self(x)
        query, key, value, edges = self.get_qkve(x, edge_attr)
        query, key, value, edges = self.shard_qkve_heads(
            query, key, value, edges, shapes, batch_size, model_comm_group
        )
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        num_chunks = self.num_chunks if self.training else int(
            os.environ.get("ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR", self.num_chunks)
        )
        attn_out = self.attention_block(query, key, value, edges, edge_index, size, num_chunks)
        attn_out = self.shard_output_seq(attn_out, shapes, batch_size, model_comm_group)
        attention_delta = torch.cat(
            [self.projection(chunk) for chunk in torch.tensor_split(attn_out + x_r, num_chunks, dim=0)],
            dim=0,
        )
        out = attention_delta + x_skip
        mlp_delta = self.run_node_dst_mlp(out, **cond_kwargs)
        nodes_new = mlp_delta + out
        meta = getattr(self, "_mechanistic_diag_meta", None)
        if ACTIVE_CONTEXT is not None and meta is not None:
            ACTIVE_CONTEXT.record_block(
                module_name=meta["module_name"],
                phase=meta["phase"],
                block_index=meta["block_index"],
                depth_order=meta["depth_order"],
                block_output=nodes_new,
                attention_delta=attention_delta,
                attention_input=x_skip,
                mlp_delta=mlp_delta,
                mlp_input=out,
            )
        return nodes_new, edge_attr

    def message(
        self,
        heads,
        query_i,
        key_j,
        value_j,
        edge_attr,
        index,
        ptr,
        size_i,
    ):
        if edge_attr is not None:
            key_j = key_j + edge_attr
        alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = dropout(alpha, p=self.dropout, training=self.training)
        meta = getattr(self, "_mechanistic_diag_meta", None)
        if ACTIVE_CONTEXT is not None and meta is not None:
            ACTIVE_CONTEXT.record_attention(
                module_name=meta["module_name"],
                alpha=alpha,
                index=index,
                size_i=size_i,
            )
        return (value_j + edge_attr) * alpha.view(-1, heads, 1)

    global ACTIVE_CONTEXT
    ACTIVE_CONTEXT = recorder
    GraphTransformerMapperBlock.forward = mapper_forward
    GraphTransformerProcessorBlock.forward = processor_forward
    GraphTransformerConv.message = message
    try:
        yield
    finally:
        ACTIVE_CONTEXT = None
        GraphTransformerMapperBlock.forward = original_mapper_forward
        GraphTransformerProcessorBlock.forward = original_processor_forward
        GraphTransformerConv.message = original_message


def _autocast_context(device: str, precision: str):
    enabled = device.startswith("cuda") and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def _get_channel_index(name_to_idx: dict[str, int], channel_name: str) -> int:
    if channel_name in name_to_idx:
        return int(name_to_idx[channel_name])
    raise KeyError(f"Missing required low-resolution input channel {channel_name!r}")


def build_mask(lons: np.ndarray, lats: np.ndarray, region_name: str) -> np.ndarray:
    if region_name not in PREDICTION_REGION_BOXES:
        raise KeyError(f"Unknown region {region_name!r}")
    lat_min, lat_max, lon_min, lon_max = PREDICTION_REGION_BOXES[region_name]
    # Normalise to -180..180 so region boxes (which use negative lons) work with
    # bundles that store lons in 0..360.
    lons_norm = np.where(np.asarray(lons) > 180.0, np.asarray(lons) - 360.0, np.asarray(lons))
    return (
        (lons_norm >= lon_min)
        & (lons_norm <= lon_max)
        & (np.asarray(lats) >= lat_min)
        & (np.asarray(lats) <= lat_max)
    )


def _apply_perturbation(
    *,
    x_lres_np: np.ndarray,
    lon_lres: np.ndarray,
    lat_lres: np.ndarray,
    channel_map: dict[str, int],
    perturbation: str,
    case: SelectedCase,
    control_region: str,
) -> tuple[np.ndarray, float]:
    x_mut = np.array(x_lres_np, copy=True)
    if perturbation == "storm_core_wind":
        region_name = case.primary_region
        channels = ["10u", "10v"]
        sign = 1.0
    elif perturbation == "storm_core_msl":
        region_name = case.primary_region
        channels = ["msl"]
        sign = -1.0
    elif perturbation == "background_wind":
        region_name = control_region
        channels = ["10u", "10v"]
        sign = 1.0
    else:
        raise ValueError(f"Unknown perturbation {perturbation!r}")
    mask = build_mask(lon_lres, lat_lres, region_name)
    if not np.any(mask):
        raise ValueError(f"Perturbation mask for {region_name} is empty.")
    for channel_name in channels:
        idx = _get_channel_index(channel_map, channel_name)
        vals = x_mut[mask, idx]
        eps = float(np.nanstd(vals, ddof=0) * 0.01)
        if not np.isfinite(eps) or eps <= 0.0:
            eps = 1.0e-3
        x_mut[mask, idx] = x_mut[mask, idx] + sign * eps
    delta_input_norm = float(np.linalg.norm((x_mut - x_lres_np).reshape(-1), ord=2))
    return x_mut, delta_input_norm


def run_prediction(
    *,
    inference_model,
    datamodule,
    device: str,
    bundle_path: str,
    sampler_args: dict[str, Any],
    precision: str,
    perturbation: str | None,
    case: SelectedCase,
    seed: int,
    control_region: str,
) -> PredictionArrays:
    name_to_idx_lres = datamodule.data_indices.data.input[0].name_to_index
    name_to_idx_hres = datamodule.data_indices.data.input[1].name_to_index
    name_to_idx_out = datamodule.data_indices.model.output.name_to_index
    _validate_bundle_hres_contract(
        bundle_nc=bundle_path,
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
    ) = load_inputs_from_bundle_numpy(bundle_path, name_to_idx_lres, name_to_idx_hres)
    if perturbation is not None:
        x_lres_np, _ = _apply_perturbation(
            x_lres_np=x_lres_np,
            lon_lres=lon_lres,
            lat_lres=lat_lres,
            channel_map=name_to_idx_lres,
            perturbation=perturbation,
            case=case,
            control_region=control_region,
        )
    x_in = torch.from_numpy(x_lres_np).to(device)[None, None, None, ...]
    x_in_hres = torch.from_numpy(x_hres_np).to(device)[None, None, None, ...]
    with torch.inference_mode():
        with _autocast_context(device=device, precision=precision):
            pred = inference_model.predict_step(
                x_in[0:1],
                x_in_hres[0:1],
                model_comm_group=None,
                extra_args={**sampler_args, "seed": int(seed)},
            )
    weather_states = list(name_to_idx_out.keys())
    pred_np = pred[0, 0, 0].detach().cpu().numpy().astype(np.float32)
    target_np, _ = extract_target_from_bundle(bundle_path, weather_states)
    return PredictionArrays(
        y_pred=pred_np,
        y_true=target_np,
        weather_states=weather_states,
        x_lres_raw=x_lres_np,
        lon_lres=np.asarray(lon_lres, dtype=np.float32),
        lat_lres=np.asarray(lat_lres, dtype=np.float32),
        lon_hres=np.asarray(lon_hres, dtype=np.float32),
        lat_hres=np.asarray(lat_hres, dtype=np.float32),
    )


def region_metrics_from_arrays(
    *,
    arrays: PredictionArrays,
    region_name: str,
) -> dict[str, float]:
    mask = build_mask(arrays.lon_hres, arrays.lat_hres, region_name)
    if not np.any(mask):
        raise ValueError(f"Primary region {region_name!r} is empty for this sample.")
    state_to_idx = {name: idx for idx, name in enumerate(arrays.weather_states)}
    u = arrays.y_pred[mask, state_to_idx["10u"]]
    v = arrays.y_pred[mask, state_to_idx["10v"]]
    msl = arrays.y_pred[mask, state_to_idx["msl"]]
    wind = np.sqrt(u**2 + v**2)
    return {
        "max_wind10m_ms": float(np.nanmax(wind)),
        "min_mslp_pa": float(np.nanmin(msl)),
        "y_pred_variance": float(np.nanvar(arrays.y_pred[mask, :])),
    }


def compute_sensitivity_rows(
    *,
    inference_model,
    datamodule,
    device: str,
    precision: str,
    sampler_args: dict[str, Any],
    case: SelectedCase,
    checkpoint_label: str,
    baseline: PredictionArrays,
    control_region: str,
    seed: int,
) -> list[dict[str, Any]]:
    if case.category == "control":
        perturbations = ["background_wind"]
    else:
        perturbations = ["storm_core_wind", "storm_core_msl"]
    base_metrics = region_metrics_from_arrays(arrays=baseline, region_name=case.primary_region)
    base_output_norm = float(np.linalg.norm(baseline.y_pred.reshape(-1), ord=2))
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        _, delta_input_norm = _apply_perturbation(
            x_lres_np=baseline.x_lres_raw,
            lon_lres=baseline.lon_lres,
            lat_lres=baseline.lat_lres,
            channel_map=datamodule.data_indices.data.input[0].name_to_index,
            perturbation=perturbation,
            case=case,
            control_region=control_region,
        )
        perturbed = run_prediction(
            inference_model=inference_model,
            datamodule=datamodule,
            device=device,
            bundle_path=case.bundle_path,
            sampler_args=sampler_args,
            precision=precision,
            perturbation=perturbation,
            case=case,
            seed=seed,
            control_region=control_region,
        )
        pert_metrics = region_metrics_from_arrays(arrays=perturbed, region_name=case.primary_region)
        delta_output_norm = float(np.linalg.norm((perturbed.y_pred - baseline.y_pred).reshape(-1), ord=2))
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "checkpoint": checkpoint_label,
                "perturbation": perturbation,
                "delta_output_over_delta_input": delta_output_norm / max(delta_input_norm, 1.0e-12),
                "delta_output_over_baseline_output": delta_output_norm / max(base_output_norm, 1.0e-12),
                "delta_max_wind10m_ms": pert_metrics["max_wind10m_ms"] - base_metrics["max_wind10m_ms"],
                "delta_min_mslp_pa": pert_metrics["min_mslp_pa"] - base_metrics["min_mslp_pa"],
            }
        )
    return rows


def evaluate_checkpoint(
    *,
    checkpoint_label: str,
    checkpoint_path: str,
    cases: list[SelectedCase],
    sampler_args: dict[str, Any],
    device: str,
    precision: str,
    validation_frequency: str,
    seed: int,
    control_region: str,
    run_sensitivity: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=checkpoint_path,
        device=device,
        validation_frequency=validation_frequency,
        precision=precision,
        num_gpus_per_model_override=1,
    )
    block_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    inference_model.eval()
    for case in cases:
        recorder = RunRecorder(case=case, checkpoint_label=checkpoint_label)
        with instrumentation_context(inference_model, recorder):
            baseline = run_prediction(
                inference_model=inference_model,
                datamodule=datamodule,
                device=device,
                bundle_path=case.bundle_path,
                sampler_args=sampler_args,
                precision=precision,
                perturbation=None,
                case=case,
                seed=seed,
                control_region=control_region,
            )
        block_rows.extend(recorder.finalize_rows())
        metrics = region_metrics_from_arrays(arrays=baseline, region_name=case.primary_region)
        case_rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "checkpoint": checkpoint_label,
                "primary_region": case.primary_region,
                "max_wind10m_ms": metrics["max_wind10m_ms"],
                "min_mslp_pa": metrics["min_mslp_pa"],
                "y_pred_variance": metrics["y_pred_variance"],
                "truth_max_wind10m_ms": case.truth_max_wind10m_ms,
                "truth_min_mslp_pa": case.truth_min_mslp_pa,
            }
        )
        if run_sensitivity and case.category in {"extreme", "control"}:
            sensitivity_rows.extend(
                compute_sensitivity_rows(
                    inference_model=inference_model,
                    datamodule=datamodule,
                    device=device,
                    precision=precision,
                    sampler_args=sampler_args,
                    case=case,
                    checkpoint_label=checkpoint_label,
                    baseline=baseline,
                    control_region=control_region,
                    seed=seed,
                )
            )
    del inference_model
    del datamodule
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        pd.DataFrame(block_rows),
        pd.DataFrame(case_rows),
        pd.DataFrame(sensitivity_rows),
    )


def build_category_summary(
    per_block_df: pd.DataFrame,
    per_case_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not per_block_df.empty:
        metrics = [
            "block_output_rms",
            "block_output_max_abs",
            "attention_delta_ratio",
            "mlp_delta_ratio",
            "attention_entropy",
            "attention_max_weight",
        ]
        grouped = per_block_df.groupby(["category", "checkpoint"], dropna=False)
        for (category, checkpoint), group in grouped:
            for metric in metrics:
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "source": "per_block",
                        "category": category,
                        "checkpoint": checkpoint,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                    }
                )
    if not per_case_df.empty:
        grouped = per_case_df.groupby(["category", "checkpoint"], dropna=False)
        for (category, checkpoint), group in grouped:
            for metric in ["max_wind10m_ms", "min_mslp_pa", "y_pred_variance"]:
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "source": "per_case",
                        "category": category,
                        "checkpoint": checkpoint,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                    }
                )
    if not sensitivity_df.empty:
        grouped = sensitivity_df.groupby(["category", "checkpoint"], dropna=False)
        for (category, checkpoint), group in grouped:
            for metric in [
                "delta_output_over_delta_input",
                "delta_max_wind10m_ms",
                "delta_min_mslp_pa",
            ]:
                if metric not in group:
                    continue
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "source": "sensitivity",
                        "category": category,
                        "checkpoint": checkpoint,
                        "metric": metric,
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                    }
                )
    return pd.DataFrame(rows)


def write_metadata(
    *,
    out_root: Path,
    args: argparse.Namespace,
    cases: list[SelectedCase],
    sampler_args: dict[str, Any],
) -> None:
    payload = {
        "checkpoint_a": args.name_ckpt_a,
        "checkpoint_b": args.name_ckpt_b,
        "input_root": args.input_root,
        "predictions_a": args.predictions_a,
        "predictions_b": args.predictions_b,
        "event": args.event,
        "control_region": args.control_region,
        "seed": int(args.seed),
        "precision": args.precision,
        "device": args.device,
        "validation_frequency": args.validation_frequency,
        "sampler_args": sampler_args,
        "run_sensitivity": not bool(args.skip_sensitivity),
        "cases": [asdict(case) for case in cases],
    }
    (out_root / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sampler_args = json.loads(args.sampler_json)
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_root).expanduser().resolve()
    predictions_a = Path(args.predictions_a).expanduser().resolve()
    predictions_b = Path(args.predictions_b).expanduser().resolve()

    if args.skip_case_selection:
        if not args.case_manifest_in:
            raise SystemExit("--skip-case-selection requires --case-manifest-in.")
        cases = load_case_manifest(Path(args.case_manifest_in).expanduser().resolve())
    else:
        cases = select_cases(
            input_root=input_root,
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            event_name=args.event,
            control_region=args.control_region,
            extreme_count=int(args.extreme_count),
            moderate_count=int(args.moderate_count),
            control_count=int(args.control_count),
        )

    save_case_manifest(cases, out_root / "case_manifest.csv")
    write_metadata(out_root=out_root, args=args, cases=cases, sampler_args=sampler_args)

    per_block_parts: list[pd.DataFrame] = []
    per_case_parts: list[pd.DataFrame] = []
    sensitivity_parts: list[pd.DataFrame] = []
    for checkpoint_label, checkpoint_path in [
        ("lowdec", args.name_ckpt_a),
        ("highdec", args.name_ckpt_b),
    ]:
        per_block_df, per_case_df, sensitivity_df = evaluate_checkpoint(
            checkpoint_label=checkpoint_label,
            checkpoint_path=checkpoint_path,
            cases=cases,
            sampler_args=sampler_args,
            device=args.device,
            precision=args.precision,
            validation_frequency=args.validation_frequency,
            seed=int(args.seed),
            control_region=args.control_region,
            run_sensitivity=not bool(args.skip_sensitivity),
        )
        per_block_parts.append(per_block_df)
        per_case_parts.append(per_case_df)
        if not sensitivity_df.empty:
            sensitivity_parts.append(sensitivity_df)

    per_block_df = pd.concat(per_block_parts, ignore_index=True) if per_block_parts else pd.DataFrame()
    per_case_df = pd.concat(per_case_parts, ignore_index=True) if per_case_parts else pd.DataFrame()
    sensitivity_df = (
        pd.concat(sensitivity_parts, ignore_index=True) if sensitivity_parts else pd.DataFrame()
    )
    category_df = build_category_summary(per_block_df, per_case_df, sensitivity_df)

    per_block_df.to_csv(out_root / "per_block_metrics.csv", index=False)
    per_case_df.to_csv(out_root / "per_case_summary.csv", index=False)
    sensitivity_df.to_csv(out_root / "sensitivity.csv", index=False)
    category_df.to_csv(out_root / "category_summary.csv", index=False)


if __name__ == "__main__":
    main()
