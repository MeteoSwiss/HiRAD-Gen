# ============================
# Sigma-eval plotter (runnable)
# - choose experiments to load
# - choose metrics to plot
# - prints available metrics
# ============================

from __future__ import annotations

import argparse

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# Config: base dir for run ids
# ----------------------------
DIR_EXP = Path("/home/ecm5702/scratch/aifs/checkpoint")


# ----------------------------
# Reader + cleaner
# ----------------------------
def clean_sigma_table(df: pd.DataFrame, csv_path: Path | str = "<in-memory>") -> pd.DataFrame:
    required = ["sigma", "prediction_on_pure_noise", "loss"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}; got {list(df.columns)}")

    df = df.copy()

    # cast boolean-ish column
    df["prediction_on_pure_noise"] = (
        df["prediction_on_pure_noise"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # numeric coercion for all others
    for c in df.columns:
        if c != "prediction_on_pure_noise":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = df[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{csv_path}: contains NaNs after coercion:\n{df.loc[bad, required]}")
    return df


def read_sigma_table_from_runfile(
    run_and_csv: str,
    *,
    dir_exp: Path = DIR_EXP,
    exp_label: str | None = None,
) -> pd.DataFrame:
    """
    run_and_csv: "RUN_ID/sigma_eval_table_XXXX.csv" (relative to dir_exp)
    """
    run_and_csv = run_and_csv.strip().lstrip("/")
    csv_path = (dir_exp / run_and_csv).resolve()

    dir_exp = dir_exp.resolve()
    if dir_exp not in csv_path.parents:
        raise ValueError(f"Refusing path outside dir_exp: {csv_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = clean_sigma_table(df, csv_path=csv_path)

    run_id = csv_path.parent.name
    df.insert(0, "exp", exp_label if exp_label is not None else run_id)
    df.insert(1, "run_id", run_id)
    df.insert(2, "source_csv", csv_path.name)
    return df


def build_all_df(specs: Sequence[str] | Sequence[tuple[str, str]], *, dir_exp: Path = DIR_EXP) -> pd.DataFrame:
    """
    Two accepted formats:
      1) list[str]:
         ["RUNID/file.csv", ...]           -> exp defaults to run_id
      2) list[tuple[str,str]]:
         [("my_label", "RUNID/file.csv")]  -> exp = my_label
    """
    if len(specs) == 0:
        raise ValueError("specs is empty")

    frames: list[pd.DataFrame] = []
    for item in specs:
        if isinstance(item, tuple):
            exp_label, run_and_csv = item
        else:
            exp_label, run_and_csv = None, item

        frames.append(
            read_sigma_table_from_runfile(
                run_and_csv,
                dir_exp=dir_exp,
                exp_label=exp_label,
            )
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["exp", "prediction_on_pure_noise", "sigma"]).reset_index(drop=True)
    return all_df


# ----------------------------
# Metrics helpers
# ----------------------------
def available_metrics(df: pd.DataFrame) -> list[str]:
    """
    Returns candidate metrics to plot (numeric columns, excluding metadata).
    """
    exclude = {"exp", "run_id", "source_csv", "prediction_on_pure_noise"}
    # keep only numeric columns
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # commonly you want sigma on x-axis, so exclude from y candidates
    y_candidates = [c for c in numeric_cols if c != "sigma"]
    return sorted(y_candidates)


def print_available_metrics(df: pd.DataFrame, *, max_show: int = 200) -> None:
    mets = available_metrics(df)
    print(f"Available metrics ({len(mets)}):")
    if len(mets) <= max_show:
        for m in mets:
            print(" -", m)
    else:
        for m in mets[:max_show]:
            print(" -", m)
        print(f"... ({len(mets) - max_show} more)")


# ----------------------------
# Plotting
# ----------------------------


def save_sigma_curves_pdf(
    all_df: pd.DataFrame,
    metrics: Sequence[str],
    pdf_path: str | Path,
    *,
    sigma_min: float = 0.02,
    pred_flags: Sequence[bool] = (False, True),
    agg: str = "mean",  # "mean" or "median"
    figsize: tuple[float, float] = (22, 10),
    title_size: int = 26,
    label_size: int = 22,
    tick_size: int = 18,
    legend_size: int = 18,
    line_width: float = 2.5,
    marker_size: float = 7.0,
    legend_outside: bool = True,
    pdf_dpi: int = 300,  # only matters for rasterized artists; vector PDF zooms well anyway
) -> Path:
    pdf_path = Path(pdf_path)

    # validate metrics
    missing = [m for m in metrics if m not in all_df.columns]
    if missing:
        raise ValueError(f"Requested metrics not in dataframe: {missing}")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        for metric in metrics:
            for pred_flag in pred_flags:
                sub = all_df[all_df["prediction_on_pure_noise"] == pred_flag]
                sub = sub[sub["sigma"] > sigma_min]

                if sub.empty:
                    print(
                        f"[WARN] No rows after filtering for metric={metric}, "
                        f"pred_flag={pred_flag}, sigma_min={sigma_min}"
                    )
                    continue

                if agg == "mean":
                    g = sub.groupby(["exp", "sigma"], as_index=False)[metric].mean()
                else:
                    g = sub.groupby(["exp", "sigma"], as_index=False)[metric].median()

                fig, ax = plt.subplots(figsize=figsize)

                for exp, gg in g.groupby("exp"):
                    gg = gg.sort_values("sigma")
                    ax.plot(
                        gg["sigma"],
                        gg[metric],
                        marker="o",
                        linewidth=line_width,
                        markersize=marker_size,
                        label=exp,
                    )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("sigma", fontsize=label_size)
                ax.set_ylabel(metric, fontsize=label_size)
                ax.set_title(
                    f"{metric} – prediction_on_pure_noise={pred_flag}, σ > {sigma_min} ({agg})",
                    fontsize=title_size,
                )
                ax.tick_params(labelsize=tick_size)

                if legend_outside:
                    ax.legend(
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        fontsize=legend_size,
                    )
                    fig.tight_layout(rect=[0, 0, 0.8, 1])
                else:
                    ax.legend(fontsize=legend_size)
                    fig.tight_layout()

                # Write this page to the PDF (vector by default)
                pdf.savefig(fig, bbox_inches="tight", dpi=pdf_dpi)
                plt.close(fig)
                n_pages += 1

        # optional: embed metadata
        d = pdf.infodict()
        d["Title"] = "Sigma evaluation curves"
        d["Creator"] = "matplotlib"

    if n_pages == 0:
        raise RuntimeError("No pages were written (all plots empty after filtering).")

    return pdf_path


# ----------------------------
# Experiment selection
# ----------------------------
def filter_specs(
    all_specs: Sequence[tuple[str, str]],
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Filter by exp label (first element of tuple).
    include/exclude are lists of labels; if include is None -> keep all.
    """
    specs = list(all_specs)

    if include is not None:
        include_set = set(include)
        specs = [s for s in specs if s[0] in include_set]

    if exclude is not None:
        exclude_set = set(exclude)
        specs = [s for s in specs if s[0] not in exclude_set]

    if not specs:
        raise ValueError("No experiments left after filtering.")
    return specs

def _parse_bool_token(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean token: {value}")


def _parse_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, run_and_csv = spec.split("=", 1)
        label = label.strip()
        run_and_csv = run_and_csv.strip()
        if not label or not run_and_csv:
            raise ValueError(f"Invalid --spec value: {spec}")
        return label, run_and_csv

    run_and_csv = spec.strip()
    if not run_and_csv:
        raise ValueError("Empty --spec value")
    run_id = Path(run_and_csv).parent.name
    return run_id, run_and_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot sigma-evaluator CSV overlays.")
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Either <label>=<run_id/file.csv> or <run_id/file.csv>. Repeat for each curve family.",
    )
    parser.add_argument(
        "--dir-exp",
        default=str(DIR_EXP),
        help="Base directory used to resolve RUNID/file.csv specs.",
    )
    parser.add_argument(
        "--metrics",
        default="loss,diff_all_var_non_weighted",
        help="Comma-separated metric columns to plot.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Output PDF path for the sigma curves.",
    )
    parser.add_argument(
        "--merged-csv",
        default="",
        help="Optional output CSV path for the merged sigma table.",
    )
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--agg", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--pred-flags",
        default="false",
        help="Comma-separated booleans for prediction_on_pure_noise, e.g. false,true",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print available metrics after loading the merged table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    specs = [_parse_spec(spec) for spec in args.spec]
    all_df = build_all_df(specs, dir_exp=Path(args.dir_exp))
    print(all_df.head())
    print("rows:", len(all_df), "exps:", all_df["exp"].nunique())

    if args.list_metrics:
        print_available_metrics(all_df)

    metrics = []
    for metric in (token.strip() for token in args.metrics.split(",")):
        if not metric:
            continue
        if metric == "diff_all_var_non_weighted" and metric not in all_df.columns:
            metric = "metric__diff_all_var_non_weighted"
        metrics.append(metric)
    if not metrics:
        raise SystemExit("No metrics requested.")

    if args.merged_csv:
        merged_csv = Path(args.merged_csv)
        merged_csv.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(merged_csv, index=False)
        print("Wrote merged CSV:", merged_csv)

    pred_flags = tuple(_parse_bool_token(token) for token in args.pred_flags.split(",") if token.strip())
    pdf_path = save_sigma_curves_pdf(
        all_df,
        metrics=metrics,
        pdf_path=args.pdf_path,
        sigma_min=args.sigma_min,
        pred_flags=pred_flags,
        agg=args.agg,
    )
    print("Wrote:", pdf_path)


if __name__ == "__main__":
    main()
