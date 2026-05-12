#!/usr/bin/env python3
"""
Plot MLflow loss curves from LOCAL logs — no server needed.

Handles long runs split across parent + resumed child runs.

USAGE:
    python plot_loss.py <experiment_dir> [name_filter] [output.png]

EXAMPLES:
    # Plot all runs
    python plot_loss.py ~/scratch/aifs/logs/mlflow/909682684414341917

    # Filter by name substring
    python plot_loss.py ~/scratch/aifs/logs/mlflow/909682684414341917 o1280

    # Save to specific file
    python plot_loss.py ~/scratch/aifs/logs/mlflow/909682684414341917 o1280 my_plot.png

WHAT IT DOES:
    - Reads metric files directly from the filesystem (no mlflow server)
    - Merges parent + child (resumed) runs into one continuous curve
    - Plots train and val loss vs step
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# ── which metrics to plot ──────────────────────────────────────────────────────
METRICS = [
    "train_weighted_mse_loss_epoch",
    "val_weighted_mse_loss_epoch",
]

# runs with this name are skipped
SKIP_NAMES = {"to_delete", ""}


# ── reading ────────────────────────────────────────────────────────────────────

def read_metric(path):
    """Read one mlflow metric file → list of (step, value).

    Each line in a local mlflow metric file is: timestamp value step
    """
    points = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                _ts, val, step = parts
                points.append((int(step), float(val)))
    return points


def load_run(run_dir):
    """Load one run directory → dict with name, parent_id, metrics."""
    run_dir = Path(run_dir)

    def read_tag(name):
        p = run_dir / "tags" / name
        return p.read_text().strip() if p.exists() else None

    name = read_tag("mlflow.runName") or run_dir.name
    parent = read_tag("mlflow.parentRunId")

    metrics = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.iterdir():
            if f.is_file():
                metrics[f.name] = read_metric(f)

    return {"id": run_dir.name, "name": name, "parent": parent, "metrics": metrics}


def load_experiment(experiment_dir):
    """Load all runs in an experiment directory → dict keyed by run_id."""
    runs = {}
    for d in Path(experiment_dir).iterdir():
        if d.is_dir() and d.name != ".trash":
            r = load_run(d)
            runs[r["id"]] = r
    return runs


# ── merging ────────────────────────────────────────────────────────────────────

def merge_family(runs, root_id):
    """Merge metrics from a root run + all its children → {metric: (steps, vals)}.

    Resumed runs create child runs with the same name. We combine them by step,
    deduplicating (last writer wins) and sorting.
    """
    family_ids = [root_id] + [rid for rid, r in runs.items() if r["parent"] == root_id]

    merged = {}
    for metric in METRICS:
        all_points = []
        for rid in family_ids:
            all_points.extend(runs[rid]["metrics"].get(metric, []))

        if not all_points:
            continue

        # deduplicate by step (keep last), then sort
        step_val = {}
        for step, val in all_points:
            step_val[step] = val
        steps, vals = zip(*sorted(step_val.items()))
        merged[metric] = (list(steps), list(vals))

    return merged


# ── plotting ───────────────────────────────────────────────────────────────────

def plot(experiment_dir, name_filter=None, output="loss_curves.png"):
    runs = load_experiment(experiment_dir)

    # collect root runs (no parent), apply filter, skip junk names, skip runs with no data
    roots = {}
    for rid, r in runs.items():
        if r["parent"] is not None:
            continue
        if r["name"] in SKIP_NAMES:
            continue
        if name_filter and name_filter not in r["name"]:
            continue
        # skip runs that have no data for any of the target metrics
        merged = merge_family(runs, rid)
        if not merged:
            continue
        roots[r["name"]] = rid

    if not roots:
        print(f"No runs found (filter={name_filter!r}). Available names:")
        for r in runs.values():
            if r["parent"] is None:
                print(f"  {r['name']}")
        return

    print(f"Found {len(roots)} run(s):")
    for name in sorted(roots):
        print(f"  {name}")

    fig, axs = plt.subplots(len(METRICS), 1, figsize=(12, 4 * len(METRICS)), sharex=True)
    if len(METRICS) == 1:
        axs = [axs]

    for name, root_id in sorted(roots.items()):
        merged = merge_family(runs, root_id)
        for ax, metric in zip(axs, METRICS):
            if metric in merged:
                steps, vals = merged[metric]
                ax.plot(steps, vals, label=name, linewidth=1.5)

    for ax, metric in zip(axs, METRICS):
        ax.set_ylabel(metric.replace("_", " "), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Step")
    exp_label = Path(experiment_dir).name
    fig.suptitle(f"Loss curves — experiment {exp_label}", fontsize=11)
    plt.tight_layout()

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {Path(output).resolve()}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    experiment_dir = sys.argv[1]
    name_filter = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else "loss_curves.png"

    plot(experiment_dir, name_filter, output)
