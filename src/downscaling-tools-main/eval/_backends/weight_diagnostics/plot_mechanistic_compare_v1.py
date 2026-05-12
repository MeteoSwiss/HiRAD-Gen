from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

CATEGORY_ORDER = ["extreme", "moderate", "control"]
CHECKPOINT_ORDER = ["lowdec", "highdec"]
CHECKPOINT_LABELS = {
    "lowdec": "wd=0.01",
    "highdec": "wd=0.1",
}
COLORS = {
    "lowdec": "#c65d00",
    "highdec": "#1f4e79",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot minimal mechanistic comparison outputs.")
    parser.add_argument("--run-root", required=True)
    return parser


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _mean_by_depth(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group = (
        df.groupby(["category", "checkpoint", "depth_order"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    return group


def _setup_category_axes(title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle(title)
    return fig, axes


def plot_activation(per_block_df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = _setup_category_axes("Block Output RMS by Depth")
    means = _mean_by_depth(per_block_df, "block_output_rms")
    for ax, category in zip(axes, CATEGORY_ORDER):
        cat = means[means["category"] == category]
        for checkpoint in CHECKPOINT_ORDER:
            sub = cat[cat["checkpoint"] == checkpoint].sort_values("depth_order")
            if sub.empty:
                continue
            ax.plot(
                sub["depth_order"],
                sub["block_output_rms"],
                marker="o",
                color=COLORS[checkpoint],
                label=CHECKPOINT_LABELS[checkpoint],
            )
        ax.set_title(category)
        ax.set_xlabel("Depth")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMS")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_residuals(per_block_df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("Residual Update Ratios by Depth")
    metrics = [
        ("attention_delta_ratio", "Attention"),
        ("mlp_delta_ratio", "MLP"),
    ]
    for row_idx, (metric, row_title) in enumerate(metrics):
        means = _mean_by_depth(per_block_df, metric)
        for col_idx, category in enumerate(CATEGORY_ORDER):
            ax = axes[row_idx, col_idx]
            cat = means[means["category"] == category]
            for checkpoint in CHECKPOINT_ORDER:
                sub = cat[cat["checkpoint"] == checkpoint].sort_values("depth_order")
                if sub.empty:
                    continue
                ax.plot(
                    sub["depth_order"],
                    sub[metric],
                    marker="o",
                    color=COLORS[checkpoint],
                    label=CHECKPOINT_LABELS[checkpoint],
                )
            ax.set_title(f"{category} | {row_title}")
            ax.grid(alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("Depth")
    axes[0, 0].set_ylabel("delta / input")
    axes[1, 0].set_ylabel("delta / input")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_attention(per_block_df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle("Attention Concentration by Depth")
    metrics = [
        ("attention_entropy", "Entropy"),
        ("attention_max_weight", "Max weight"),
    ]
    for row_idx, (metric, row_title) in enumerate(metrics):
        means = _mean_by_depth(per_block_df, metric)
        for col_idx, category in enumerate(CATEGORY_ORDER):
            ax = axes[row_idx, col_idx]
            cat = means[means["category"] == category]
            for checkpoint in CHECKPOINT_ORDER:
                sub = cat[cat["checkpoint"] == checkpoint].sort_values("depth_order")
                if sub.empty:
                    continue
                ax.plot(
                    sub["depth_order"],
                    sub[metric],
                    marker="o",
                    color=COLORS[checkpoint],
                    label=CHECKPOINT_LABELS[checkpoint],
                )
            ax.set_title(f"{category} | {row_title}")
            ax.grid(alpha=0.3)
            if row_idx == 1:
                ax.set_xlabel("Depth")
    axes[0, 0].set_ylabel("Entropy")
    axes[1, 0].set_ylabel("Mean max weight")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_sensitivity(sensitivity_df: pd.DataFrame, out_path: Path) -> Path:
    fig, axes = _setup_category_axes("Finite-Difference Sensitivity")
    summary = (
        sensitivity_df.groupby(["category", "checkpoint", "perturbation"], dropna=False)["delta_output_over_delta_input"]
        .mean()
        .reset_index()
    )
    for ax, category in zip(axes, CATEGORY_ORDER):
        cat = summary[summary["category"] == category]
        perturbations = list(cat["perturbation"].drop_duplicates())
        if not perturbations:
            ax.set_title(category)
            ax.axis("off")
            continue
        x = np.arange(len(perturbations))
        width = 0.35
        for idx, checkpoint in enumerate(CHECKPOINT_ORDER):
            sub = cat[cat["checkpoint"] == checkpoint]
            vals = [
                float(sub[sub["perturbation"] == perturbation]["delta_output_over_delta_input"].mean())
                if not sub[sub["perturbation"] == perturbation].empty
                else np.nan
                for perturbation in perturbations
            ]
            ax.bar(x + (idx - 0.5) * width, vals, width=width, color=COLORS[checkpoint], label=CHECKPOINT_LABELS[checkpoint])
        ax.set_xticks(x, perturbations, rotation=20, ha="right")
        ax.set_title(category)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("||delta output|| / ||delta input||")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _metric_mean(category_df: pd.DataFrame, category: str, checkpoint: str, metric: str) -> float | None:
    sub = category_df[
        (category_df["category"] == category)
        & (category_df["checkpoint"] == checkpoint)
        & (category_df["metric"] == metric)
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["mean"])


def build_summary_text(category_df: pd.DataFrame, per_case_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> str:
    lines: list[str] = ["# Mechanistic Compare V1", ""]
    extreme_low_rms = _metric_mean(category_df, "extreme", "lowdec", "block_output_rms")
    extreme_high_rms = _metric_mean(category_df, "extreme", "highdec", "block_output_rms")
    extreme_low_entropy = _metric_mean(category_df, "extreme", "lowdec", "attention_entropy")
    extreme_high_entropy = _metric_mean(category_df, "extreme", "highdec", "attention_entropy")
    extreme_low_sens = _metric_mean(category_df, "extreme", "lowdec", "delta_output_over_delta_input")
    extreme_high_sens = _metric_mean(category_df, "extreme", "highdec", "delta_output_over_delta_input")

    lines.append("## Readout")
    if extreme_low_rms is not None and extreme_high_rms is not None:
        if extreme_low_rms > extreme_high_rms:
            lines.append("- Extreme cases: the low-decay checkpoint has higher mean block-output RMS.")
        elif extreme_low_rms < extreme_high_rms:
            lines.append("- Extreme cases: the high-decay checkpoint has higher mean block-output RMS.")
        else:
            lines.append("- Extreme cases: the checkpoints are tied on mean block-output RMS.")
    if extreme_low_entropy is not None and extreme_high_entropy is not None:
        if extreme_low_entropy < extreme_high_entropy:
            lines.append("- Extreme cases: the low-decay checkpoint has lower attention entropy, which is consistent with sharper focus.")
        elif extreme_low_entropy > extreme_high_entropy:
            lines.append("- Extreme cases: the high-decay checkpoint has lower attention entropy.")
    if extreme_low_sens is not None and extreme_high_sens is not None:
        if extreme_low_sens > extreme_high_sens:
            lines.append("- Extreme cases: the low-decay checkpoint is more sensitive under the finite-difference probe.")
        elif extreme_low_sens < extreme_high_sens:
            lines.append("- Extreme cases: the high-decay checkpoint is more sensitive under the finite-difference probe.")

    lines.append("")
    lines.append("## Caveat")
    lines.append("- This summary is associational only. It compares matched inference runs and lightweight internal diagnostics; it does not prove a causal mechanism.")

    if not per_case_df.empty:
        lines.append("")
        lines.append("## Per-Case Output Link")
        pivot = per_case_df.pivot_table(index=["case_id", "category"], columns="checkpoint", values=["max_wind10m_ms", "min_mslp_pa", "y_pred_variance"], aggfunc="first")
        for (case_id, category), row in pivot.iterrows():
            lines.append(f"- {case_id} ({category})")
            for metric in ["max_wind10m_ms", "min_mslp_pa", "y_pred_variance"]:
                low = row.get((metric, "lowdec"))
                high = row.get((metric, "highdec"))
                if pd.notna(low) and pd.notna(high):
                    lines.append(f"  - {metric}: lowdec={float(low):.4f}, highdec={float(high):.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_block_df = _load_csv(run_root / "per_block_metrics.csv")
    per_case_df = _load_csv(run_root / "per_case_summary.csv")
    sensitivity_df = _load_csv(run_root / "sensitivity.csv")
    category_df = _load_csv(run_root / "category_summary.csv")

    if not per_block_df.empty:
        plot_activation(per_block_df, plots_dir / "activation_rms_vs_depth.png")
        plot_residuals(per_block_df, plots_dir / "residual_ratio_vs_depth.png")
        plot_attention(per_block_df, plots_dir / "attention_entropy_vs_depth.png")
    if not sensitivity_df.empty:
        plot_sensitivity(sensitivity_df, plots_dir / "sensitivity_compare.png")

    summary_text = build_summary_text(category_df, per_case_df, sensitivity_df)
    (run_root / "summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
