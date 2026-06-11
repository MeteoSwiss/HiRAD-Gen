from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from eval._backends.weight_diagnostics import mechanistic_compare_v1 as run_mod
from eval._backends.weight_diagnostics import plot_mechanistic_compare_v1 as plot_mod


def test_tensor_helpers_and_attention_accumulator():
    tensor = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32)
    assert run_mod.tensor_max_abs(tensor) == 4.0
    assert run_mod.tensor_ratio(tensor, tensor) == 1.0
    assert run_mod.tensor_rms(tensor) > 0.0

    acc = run_mod.AttentionAccumulator()
    alpha = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float32)
    index = torch.tensor([0, 0], dtype=torch.long)
    acc.update(alpha=alpha, index=index, size_i=1)
    metrics = acc.as_metrics()
    assert 0.0 < metrics["attention_entropy"] < 1.0
    assert 0.0 < metrics["attention_max_weight"] <= 1.0


def test_build_category_summary_aggregates_expected_metrics():
    per_block_df = pd.DataFrame(
        [
            {
                "case_id": "extreme_a",
                "category": "extreme",
                "checkpoint": "lowdec",
                "block_output_rms": 2.0,
                "block_output_max_abs": 3.0,
                "attention_delta_ratio": 1.2,
                "mlp_delta_ratio": 0.8,
                "attention_entropy": 0.4,
                "attention_max_weight": 0.7,
            },
            {
                "case_id": "extreme_a",
                "category": "extreme",
                "checkpoint": "highdec",
                "block_output_rms": 1.0,
                "block_output_max_abs": 2.0,
                "attention_delta_ratio": 0.6,
                "mlp_delta_ratio": 0.4,
                "attention_entropy": 0.9,
                "attention_max_weight": 0.4,
            },
        ]
    )
    per_case_df = pd.DataFrame(
        [
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "lowdec", "max_wind10m_ms": 30.0, "min_mslp_pa": 98000.0, "y_pred_variance": 4.0},
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "highdec", "max_wind10m_ms": 24.0, "min_mslp_pa": 98600.0, "y_pred_variance": 2.5},
        ]
    )
    sensitivity_df = pd.DataFrame(
        [
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "lowdec", "delta_output_over_delta_input": 1.5, "delta_max_wind10m_ms": 0.7, "delta_min_mslp_pa": -40.0},
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "highdec", "delta_output_over_delta_input": 0.9, "delta_max_wind10m_ms": 0.2, "delta_min_mslp_pa": -10.0},
        ]
    )

    summary = run_mod.build_category_summary(per_block_df, per_case_df, sensitivity_df)
    assert not summary.empty
    assert {"per_block", "per_case", "sensitivity"}.issubset(set(summary["source"]))


def test_plotter_writes_outputs_and_summary(tmp_path: Path):
    per_block_df = pd.DataFrame(
        [
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "lowdec", "depth_order": 0, "block_output_rms": 2.0, "block_output_max_abs": 3.0, "attention_delta_ratio": 1.1, "mlp_delta_ratio": 0.8, "attention_entropy": 0.4, "attention_max_weight": 0.7},
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "highdec", "depth_order": 0, "block_output_rms": 1.2, "block_output_max_abs": 2.2, "attention_delta_ratio": 0.7, "mlp_delta_ratio": 0.5, "attention_entropy": 0.8, "attention_max_weight": 0.5},
            {"case_id": "control_a", "category": "control", "checkpoint": "lowdec", "depth_order": 17, "block_output_rms": 1.0, "block_output_max_abs": 1.5, "attention_delta_ratio": 0.4, "mlp_delta_ratio": 0.3, "attention_entropy": 0.6, "attention_max_weight": 0.5},
            {"case_id": "control_a", "category": "control", "checkpoint": "highdec", "depth_order": 17, "block_output_rms": 0.9, "block_output_max_abs": 1.4, "attention_delta_ratio": 0.35, "mlp_delta_ratio": 0.25, "attention_entropy": 0.7, "attention_max_weight": 0.45},
        ]
    )
    per_case_df = pd.DataFrame(
        [
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "lowdec", "max_wind10m_ms": 30.0, "min_mslp_pa": 98000.0, "y_pred_variance": 4.0},
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "highdec", "max_wind10m_ms": 24.0, "min_mslp_pa": 98600.0, "y_pred_variance": 2.5},
        ]
    )
    sensitivity_df = pd.DataFrame(
        [
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "lowdec", "perturbation": "storm_core_wind", "delta_output_over_delta_input": 1.6},
            {"case_id": "extreme_a", "category": "extreme", "checkpoint": "highdec", "perturbation": "storm_core_wind", "delta_output_over_delta_input": 0.8},
        ]
    )
    category_df = run_mod.build_category_summary(per_block_df, per_case_df, sensitivity_df)

    run_root = tmp_path
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_mod.plot_activation(per_block_df, plots_dir / "activation_rms_vs_depth.png")
    plot_mod.plot_residuals(per_block_df, plots_dir / "residual_ratio_vs_depth.png")
    plot_mod.plot_attention(per_block_df, plots_dir / "attention_entropy_vs_depth.png")
    plot_mod.plot_sensitivity(sensitivity_df, plots_dir / "sensitivity_compare.png")
    summary_text = plot_mod.build_summary_text(category_df, per_case_df, sensitivity_df)
    (run_root / "summary.md").write_text(summary_text, encoding="utf-8")

    assert (plots_dir / "activation_rms_vs_depth.png").exists()
    assert (plots_dir / "residual_ratio_vs_depth.png").exists()
    assert (plots_dir / "attention_entropy_vs_depth.png").exists()
    assert (plots_dir / "sensitivity_compare.png").exists()
    assert "low-decay checkpoint has higher mean block-output RMS" in summary_text
