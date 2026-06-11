"""
loader.py — read local MLflow logs into plain dicts. No mlflow server needed.

METRIC LAYOUT ON DISK
---------------------
  metrics/<name>                    flat file    e.g. train_weighted_mse_loss_epoch
  metrics/<group>/<var>/1           nested file  e.g. val_mse_metric/sfc_10u/1

Each file: one line per logged value   →   "<timestamp_ms> <value> <step>"

PUBLIC API
----------
  runs = load(experiment_dir)
      → dict  run_name → RunData

  runs = load_run_family(experiment_dir, run_id)
      → dict  run_name → RunData  for one requested run family, bypassing
        experiment-wide name filters

  RunData keys
      "metrics"   dict  metric_key → {"steps": [...], "vals": [...]}
      "max_step"  int   largest step logged across all metrics
      "run_id"    str

  Metric keys follow the path relative to metrics/:
      "train_weighted_mse_loss_epoch"
      "val_weighted_mse_loss_epoch"
      "lr-AdamW"
      "val_mse_metric/sfc_10u"
      "val_mse_metric/z_500"
      ...

FILTERING HELPERS
-----------------
  runs = filter_runs(runs, min_steps=50_000)
  runs = filter_runs(runs, name_contains="o320")
"""

from pathlib import Path

# ─── low-level file reading ────────────────────────────────────────────────────

def _read_metric_file(path):
    """Read one MLflow metric file → list of (step, value)."""
    points = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                _ts, val, step = parts
                points.append((int(step), float(val)))
    return points


def _dedup_and_sort(points):
    """Dedup by step (last writer wins), sort → (steps_list, vals_list)."""
    step_val = {}
    for step, val in points:
        step_val[step] = val
    if not step_val:
        return [], []
    steps, vals = zip(*sorted(step_val.items()))
    return list(steps), list(vals)


# ─── single-run loading ────────────────────────────────────────────────────────

def _read_tag(run_dir, tag_name):
    p = run_dir / "tags" / tag_name
    return p.read_text().strip() if p.exists() else None


def _load_metrics_dir(metrics_dir, prefix=""):
    """Recursively walk metrics dir, return {metric_key: [(step, val), ...]}."""
    raw = {}
    if not metrics_dir.exists():
        return raw
    for entry in metrics_dir.iterdir():
        if entry.name == "system":           # skip system metrics
            continue
        if entry.is_file():
            key = f"{prefix}{entry.name}" if prefix else entry.name
            raw[key] = _read_metric_file(entry)
        elif entry.is_dir():
            sub_prefix = f"{prefix}{entry.name}/" if prefix else f"{entry.name}/"
            raw.update(_load_metrics_dir(entry, prefix=sub_prefix))
    return raw


def _load_single_run(run_dir):
    run_dir = Path(run_dir)
    name   = _read_tag(run_dir, "mlflow.runName") or run_dir.name
    parent = _read_tag(run_dir, "mlflow.parentRunId")
    raw    = _load_metrics_dir(run_dir / "metrics")
    return {"id": run_dir.name, "name": name, "parent": parent, "raw": raw}


# ─── family merging ────────────────────────────────────────────────────────────

def _collect_descendants(all_runs_raw, root_id):
    """BFS: collect root + all transitive descendants present in all_runs_raw."""
    family = []
    queue = [root_id]
    while queue:
        current = queue.pop(0)
        if current in family:
            continue
        family.append(current)
        children = [rid for rid, r in all_runs_raw.items()
                    if r["parent"] == current and rid not in family]
        queue.extend(children)
    return family


def _merge_family(all_runs_raw, root_id):
    """Combine a root run + all transitive descendants into one RunData dict."""
    family = _collect_descendants(all_runs_raw, root_id)

    # gather all (step, val) points per metric across all family members
    combined_raw = {}
    for rid in family:
        for key, points in all_runs_raw[rid]["raw"].items():
            combined_raw.setdefault(key, []).extend(points)

    # deduplicate and sort
    metrics = {}
    for key, points in combined_raw.items():
        steps, vals = _dedup_and_sort(points)
        if steps:
            # strip trailing "/" from nested keys e.g. "val_mse_metric/sfc_10u/"
            clean_key = key.rstrip("/")
            metrics[clean_key] = {"steps": steps, "vals": vals}

    max_step = max(
        (s for m in metrics.values() for s in m["steps"]),
        default=0,
    )

    return {
        "metrics": metrics,
        "max_step": max_step,
        "run_id": root_id,
    }


# ─── experiment loading ────────────────────────────────────────────────────────

_SKIP_NAMES = {"to_delete", "to_delete_quarantine", ""}

def _looks_like_uuid(name):
    """True if name looks like a raw UUID (no human-readable parts)."""
    import re
    return bool(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", name))


def load(experiment_dir):
    """Load all runs in an experiment dir.

    Returns dict  run_name → RunData  for root runs only (children merged in).
    Skips junk names and runs with no logged metrics.
    """
    experiment_dir = Path(experiment_dir)

    # load every run directory
    all_raw = {}
    for d in experiment_dir.rglob("meta.yaml"):
        run_dir = d.parent
        # skip experiment-level meta.yaml (no tags/ sibling)
        if not (run_dir / "tags").exists() and not (run_dir / "metrics").exists():
            continue
        r = _load_single_run(run_dir)
        all_raw[r["id"]] = r

    # build output: only root runs, merge children in
    runs = {}
    for rid, r in all_raw.items():
        if r["parent"] is not None:
            continue                    # child — will be merged into parent
        name = r["name"]
        if name in _SKIP_NAMES or _looks_like_uuid(name):
            continue
        merged = _merge_family(all_raw, rid)
        if not merged["metrics"]:
            continue                    # empty ghost run
        runs[name] = merged

    return runs


def load_run_family(experiment_dir, run_id):
    """Load one run family by run_id, bypassing experiment-wide name filters.

    Walks up to the ultimate root ancestor, then BFS-collects all transitive
    descendants so the full training history is merged (handles deep walltime
    chains where each continuation is a child of the previous child).
    """
    experiment_dir = Path(experiment_dir)
    run_dir = experiment_dir / run_id
    if not run_dir.is_dir():
        return {}

    target = _load_single_run(run_dir)

    # Walk up to ultimate root (max 50 hops to prevent cycles)
    current_id = target["id"]
    current = target
    for _ in range(50):
        pid = current["parent"]
        if not pid:
            break
        parent_dir = experiment_dir / pid
        if not parent_dir.is_dir():
            break
        current = _load_single_run(parent_dir)
        current_id = pid

    root_id = current_id
    root = current

    # Build parent → children map by scanning experiment dir once
    parent_to_children = {}
    for child_dir in experiment_dir.iterdir():
        if not child_dir.is_dir() or child_dir.name == root_id:
            continue
        child_parent = _read_tag(child_dir, "mlflow.parentRunId")
        if child_parent:
            parent_to_children.setdefault(child_parent, []).append(child_dir.name)

    # BFS from root to collect all descendants
    all_raw = {root_id: root}
    queue = [root_id]
    while queue:
        cur = queue.pop(0)
        for child_id in parent_to_children.get(cur, []):
            if child_id not in all_raw:
                all_raw[child_id] = _load_single_run(experiment_dir / child_id)
                queue.append(child_id)

    merged = _merge_family(all_raw, root_id)
    if not merged["metrics"]:
        return {}

    return {root["name"] or root_id: merged}


# ─── filtering helpers ─────────────────────────────────────────────────────────

def filter_runs(runs, min_steps=0, name_contains=None):
    """Return a filtered subset of the runs dict.

    Args:
        runs:          output of load()
        min_steps:     keep only runs whose max_step >= this value
        name_contains: keep only runs whose name contains this substring
    """
    out = {}
    for name, data in runs.items():
        if data["max_step"] < min_steps:
            continue
        if name_contains and name_contains not in name:
            continue
        out[name] = data
    return out


def list_vars(runs):
    """Print all per-variable metric keys found across all runs."""
    keys = set()
    for data in runs.values():
        for k in data["metrics"]:
            if "/" in k:
                keys.add(k)
    for k in sorted(keys):
        print(k)


# ─── quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = load(exp_dir)
    runs = filter_runs(runs, min_steps=50_000)
    print(f"\n{len(runs)} run(s) with >50k steps:\n")
    for name, d in sorted(runs.items(), key=lambda x: -x[1]["max_step"]):
        print(f"  {d['max_step']:>8,}  {name}")
    print("\nPer-variable metrics available:")
    list_vars(runs)
