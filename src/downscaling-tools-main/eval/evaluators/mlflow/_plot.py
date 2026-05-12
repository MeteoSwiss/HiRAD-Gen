"""
plot.py — flexible loss curve plots from loader.py RunData dicts.

TWO MAIN PLOTS
--------------
  plot_key_vars(runs)   6-panel grid: one subplot per key surface variable
                        shows val MSE per var, all runs overlaid
                        + aggregate train/val loss as a summary panel on top

  plot_overview(runs)   3-panel: aggregate train+val, LR schedule, val-all MSE

CUSTOMIZE AT THE TOP
--------------------
  KEY_VARS   — which 6 vars get the big grid (change freely)
  VAR_LABEL  — human-readable axis labels for each var
  TRAIN_KEY / VAL_KEY — which aggregate metrics to use for the overview

STANDALONE USAGE
----------------
  python plot.py <experiment_dir> [min_steps] [name_filter]

  e.g.  python plot.py ~/scratch/aifs/logs/mlflow/909682684414341917 50000 o320
"""

from pathlib import Path
import math
import sys
import matplotlib.pyplot as plt

# ─── user-configurable constants ──────────────────────────────────────────────

KEY_VARS = [
    "val_mse_metric/sfc_10u/1",
    "val_mse_metric/sfc_10v/1",
    "val_mse_metric/sfc_2t/1",
    "val_mse_metric/sfc_tp/1",     # may be absent in some runs → skipped silently
    "val_mse_metric/sfc_msl/1",
    "val_mse_metric/z_500/1",
]

VAR_LABEL = {
    "val_mse_metric/sfc_10u/1":  "10u  (val MSE)",
    "val_mse_metric/sfc_10v/1":  "10v  (val MSE)",
    "val_mse_metric/sfc_2t/1":   "2t   (val MSE)",
    "val_mse_metric/sfc_tp/1":   "tp   (val MSE)",
    "val_mse_metric/sfc_msl/1":  "msl  (val MSE)",
    "val_mse_metric/z_500/1":    "z500 (val MSE)",
    # downscaling output metrics — shown in overview if present
    "val_out_hres_mse_metric/sfc_10u/1_scale_0": "10u hres (val MSE)",
    "val_out_hres_mse_metric/z_500/1_scale_0":   "z500 hres (val MSE)",
}

TRAIN_KEY = "train_weighted_mse_loss_epoch"
VAL_KEY   = "val_weighted_mse_loss_epoch"
LR_KEY    = "lr-AdamW"
VAL_ALL   = "val_mse_metric/all/1"

# Set True to use log scale on all per-variable val MSE panels.
# Useful when runs have very different absolute scales (e.g. mixed resolutions).
LOG_SCALE_VARS = False


# ─── color palette ────────────────────────────────────────────────────────────

def _palette(n):
    """n distinct colors from a colormap."""
    cmap = plt.colormaps["tab10" if n <= 10 else "tab20"]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _color_map(runs):
    """Return {run_name: color} dict, consistent across all plots."""
    names = sorted(runs.keys())
    palette = _palette(len(names))
    return dict(zip(names, palette))


# ─── core drawing helper ──────────────────────────────────────────────────────

def _plot_metric(ax, runs, metric_key, colors_by_name, label=True):
    """Plot one metric for all runs on ax. Returns True if anything was drawn."""
    drawn = False
    for name, data in sorted(runs.items()):
        m = data["metrics"].get(metric_key)
        if m is None:
            continue
        ax.plot(m["steps"], m["vals"], color=colors_by_name[name],
                linewidth=1.4, label=name if label else None)
        drawn = True
    return drawn


def _style(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def _add_figure_legend(fig, colors_by_name, ncol=3):
    """Single shared legend below the figure."""
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, label=n)
        for n, c in sorted(colors_by_name.items())
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               fontsize=7.5, framealpha=0.8,
               bbox_to_anchor=(0.5, -0.02))


def _all_val_metric_keys(runs):
    """Return all per-variable validation MSE keys, excluding the aggregate key."""
    keys = set()
    for data in runs.values():
        for key in data["metrics"]:
            if not key.startswith("val_mse_metric/"):
                continue
            if key == VAL_ALL:
                continue
            keys.add(key)
    return sorted(keys)


# ─── plot 1: key variable panels ──────────────────────────────────────────────

def plot_key_vars(runs, output="key_vars.png"):
    """
    Big grid: top row = aggregate train+val, bottom rows = 6 key-var val MSE.

    Layout  (3 cols × 3 rows):
      row 0:  [aggregate train+val (wide)]  [lr schedule]
      row 1:  [10u]  [10v]  [2t]
      row 2:  [tp]   [msl]  [z500]

    Single shared legend below the figure — colors consistent across all panels.
    """
    colors  = _color_map(runs)
    n_vars  = len(KEY_VARS)
    n_cols  = 3
    n_rows  = 1 + (n_vars + n_cols - 1) // n_cols   # 1 header row + var rows

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    # ── row 0: aggregate summary (spans 2 cols) + LR (1 col) ─────────────────
    ax_agg = fig.add_subplot(n_rows, n_cols, (1, 2))   # spans cols 1-2
    ax_lr  = fig.add_subplot(n_rows, n_cols, 3)

    for name, data in sorted(runs.items()):
        color = colors[name]
        t = data["metrics"].get(TRAIN_KEY)
        v = data["metrics"].get(VAL_KEY)
        if t:
            ax_agg.plot(t["steps"], t["vals"], color=color, linewidth=1.2,
                        linestyle="--", alpha=0.65)
        if v:
            ax_agg.plot(v["steps"], v["vals"], color=color, linewidth=1.5)

    _style(ax_agg, "weighted MSE loss", title="Aggregate loss  (-- train, — val)")

    _plot_metric(ax_lr, runs, LR_KEY, colors, label=False)
    _style(ax_lr, "learning rate", title="LR schedule")
    ax_lr.set_yscale("log")

    # ── rows 1+: per-variable val MSE ─────────────────────────────────────────
    for i, var_key in enumerate(KEY_VARS):
        row = 1 + i // n_cols
        col = 1 + i % n_cols
        ax  = fig.add_subplot(n_rows, n_cols, row * n_cols + col)

        drawn = _plot_metric(ax, runs, var_key, colors, label=False)
        ylabel = VAR_LABEL.get(var_key, var_key.split("/")[-2])
        _style(ax, ylabel)
        if LOG_SCALE_VARS and drawn:
            ax.set_yscale("log")

        if not drawn:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10)

    # ── shared elements ───────────────────────────────────────────────────────
    fig.supxlabel("Training step", fontsize=9, y=0.02)
    fig.suptitle("Key variable validation MSE", fontsize=12)
    _add_figure_legend(fig, colors, ncol=min(len(runs), 4))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {Path(output).resolve()}")
    plt.close()


# ─── plot 2: overview ─────────────────────────────────────────────────────────

def plot_overview(runs, output="overview.png"):
    """
    3-panel overview (+ optional hres panel for downscaling runs):
      (1) aggregate train (dashed) + val (solid) on same axes
      (2) val_mse_metric/all  — overall val MSE
      (3) LR schedule
      (4) [optional] val_out_hres_mse_metric/all — downscaling output MSE
    """
    colors = _color_map(runs)

    # check if hres metrics exist in any run
    has_hres = any(
        k.startswith("val_out_hres_mse_metric")
        for data in runs.values()
        for k in data["metrics"]
    )

    n_panels = 3 + (1 if has_hres else 0)
    fig, axs = plt.subplots(n_panels, 1, figsize=(13, 3.5 * n_panels), sharex=False)

    # panel 0: aggregate train + val
    for name, data in sorted(runs.items()):
        color = colors[name]
        t = data["metrics"].get(TRAIN_KEY)
        v = data["metrics"].get(VAL_KEY)
        if t:
            axs[0].plot(t["steps"], t["vals"], color=color, linestyle="--",
                        linewidth=1.2, alpha=0.65)
        if v:
            axs[0].plot(v["steps"], v["vals"], color=color, linewidth=1.5)
    _style(axs[0], "weighted MSE loss", title="Aggregate loss  (-- train, — val)")

    # panel 1: val_mse_metric/all  — log scale avoids one run crushing the others
    _plot_metric(axs[1], runs, VAL_ALL, colors, label=False)
    _style(axs[1], "val MSE (all vars)", title="Overall val MSE")
    axs[1].set_yscale("log")

    # panel 2: LR
    _plot_metric(axs[2], runs, LR_KEY, colors, label=False)
    _style(axs[2], "learning rate", title="LR schedule")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Training step", fontsize=9)

    # panel 3: hres MSE (downscaling) if present
    if has_hres:
        hres_all = "val_out_hres_mse_metric/all/1_scale_0"
        _plot_metric(axs[3], runs, hres_all, colors, label=False)
        _style(axs[3], "hres val MSE (all vars)", title="Downscaling output val MSE")
        axs[3].set_xlabel("Training step", fontsize=9)

    fig.suptitle("Training overview", fontsize=12)
    _add_figure_legend(fig, colors, ncol=min(len(runs), 4))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {Path(output).resolve()}")
    plt.close()


def plot_all_vars(runs, output="all_vars.png"):
    """Plot every available per-variable validation MSE in one shared grid."""
    metric_keys = _all_val_metric_keys(runs)
    if not metric_keys:
        raise ValueError("No per-variable validation MSE metrics found.")

    colors = _color_map(runs)
    n_vars = len(metric_keys)
    n_cols = min(4, n_vars)
    n_rows = math.ceil(n_vars / n_cols)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        sharex=False,
        squeeze=False,
    )

    for ax, metric_key in zip(axs.flat, metric_keys):
        drawn = _plot_metric(ax, runs, metric_key, colors, label=False)
        ylabel = VAR_LABEL.get(metric_key, metric_key.removeprefix("val_mse_metric/").replace("/1", ""))
        _style(ax, ylabel)
        if LOG_SCALE_VARS and drawn:
            ax.set_yscale("log")
        if not drawn:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )

    for ax in axs.flat[n_vars:]:
        ax.axis("off")

    fig.supxlabel("Training step", fontsize=9, y=0.02)
    fig.suptitle("All variable validation MSE", fontsize=12)
    _add_figure_legend(fig, colors, ncol=min(len(runs), 4))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {Path(output).resolve()}")
    plt.close()


# ─── convenience: plot all ────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path.home() / "perm" / "training_logs_lots"


def plot_all(runs, output_dir=None):
    """Generate the standard MLflow plot bundle. Call this from scripts or notebooks."""
    d = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    plot_key_vars(runs, output=d / "key_vars.png")
    plot_overview(runs, output=d / "overview.png")
    plot_all_vars(runs, output=d / "all_vars.png")


# ─── standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    # lazy import so the module is usable without loader on sys.path
    sys.path.insert(0, str(Path(__file__).parent))
    from loader import load, filter_runs

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir    = sys.argv[1]
    min_steps  = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    name_filt  = sys.argv[3] if len(sys.argv) > 3 else None

    runs = load(exp_dir)
    runs = filter_runs(runs, min_steps=min_steps, name_contains=name_filt)

    print(f"\nPlotting {len(runs)} run(s):")
    for name, d in sorted(runs.items(), key=lambda x: -x[1]["max_step"]):
        print(f"  {d['max_step']:>8,}  {name}")
    print()

    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    plot_all(runs, output_dir=output_dir)
