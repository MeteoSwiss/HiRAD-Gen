"""
mlflow_import.py — discover, transfer, and plot MLflow training loss for a checkpoint.

Standard entry point for all evaluation lanes.  Given a checkpoint path and
a run root, this script:

  Phase A — extracts the MLflow run UUID from the checkpoint directory name
            and reads embedded metadata to determine training origin
  Phase B — locates the MLflow logs (local or remote), rsyncing from Jupiter
            when necessary
  Phase C — loads metrics, generates plots, stores raw data

CLI:
    python mlflow_import.py \\
        --checkpoint-path /home/ecm5702/scratch/aifs/checkpoint/<UUID>/last.ckpt \\
        --run-root /home/ecm5702/scratch/eval/<RUN_ID>/ \\
        [--local-only] [--dry-run] [--force]

Exit codes:
    0 = success (or already generated)
    1 = checkpoint not found
    2 = MLflow logs not found
    3 = plot failure
    4 = rsync failure
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOS_EXPERIMENT_ID = "909682684414341917"
JUPITER_EXPERIMENT_ID = "865455718239814337"

ATOS_MLFLOW_ROOT = Path.home() / "scratch/aifs/logs/mlflow" / ATOS_EXPERIMENT_ID
JUPITER_MIRROR_ROOT = Path.home() / "perm/mlflow_mirror" / JUPITER_EXPERIMENT_ID
JUPITER_REMOTE_BASE = (
    "/e/data1/jureap-data/ecmwf/users/jdumont/outputs/logs/mlflow"
    f"/{JUPITER_EXPERIMENT_ID}"
)
JUPITER_SSH_HOST = "jupiter"

# ---------------------------------------------------------------------------
# Phase A — discover run_id and training origin
# ---------------------------------------------------------------------------


def _extract_uuid(checkpoint_path: Path) -> str:
    """Return the 32-char hex UUID from the checkpoint parent directory."""
    parent = checkpoint_path.parent.name
    if len(parent) == 32 and all(c in "0123456789abcdef" for c in parent):
        return parent
    raise SystemExit(
        f"Cannot extract UUID from checkpoint path: {checkpoint_path}\n"
        f"Expected parent dir to be a 32-char hex UUID, got: {parent}"
    )


def _read_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """Read the embedded anemoi.json metadata from a .ckpt zip archive."""
    with zipfile.ZipFile(checkpoint_path) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("anemoi-metadata/anemoi.json")]
        if not candidates:
            return {}
        return json.loads(zf.read(candidates[0]))


def _detect_training_origin(metadata: dict) -> str:
    """Determine where the checkpoint was trained based on embedded paths."""
    cfg = metadata.get("config", {})
    hw = cfg.get("hardware", {})
    paths = hw.get("paths", {})
    output = paths.get("output", "")
    checkpoints = paths.get("checkpoints", "")
    probe = f"{output} {checkpoints}"

    if "/e/data1" in probe or "jureap" in probe:
        return "jupiter"
    if "/leonardo_work" in probe or "/leonardo" in probe:
        return "leonardo"
    # ATOS / AG share the same filesystem
    return "atos"


# ---------------------------------------------------------------------------
# Phase B — locate & transfer MLflow logs
# ---------------------------------------------------------------------------


def _find_local_mlflow_dir(run_id: str) -> Path | None:
    """Check local ATOS experiment dir for the run."""
    candidate = ATOS_MLFLOW_ROOT / run_id
    if candidate.is_dir() and (candidate / "metrics").exists():
        return ATOS_MLFLOW_ROOT
    return None


def _find_jupiter_mirror(run_id: str) -> Path | None:
    """Check if Jupiter MLflow logs are already mirrored locally."""
    candidate = JUPITER_MIRROR_ROOT / run_id
    if candidate.is_dir() and (candidate / "metrics").exists():
        return JUPITER_MIRROR_ROOT
    return None


def _rsync_jupiter(run_id: str, dry_run: bool = False) -> Path:
    """Rsync a Jupiter MLflow run (and children) to the local mirror."""
    local_dest = JUPITER_MIRROR_ROOT / run_id
    remote_src = f"{JUPITER_SSH_HOST}:{JUPITER_REMOTE_BASE}/{run_id}/"

    # Verify SSH control socket
    check = subprocess.run(
        ["ssh", "-O", "check", JUPITER_SSH_HOST],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        print(f"SSH control socket not available for {JUPITER_SSH_HOST}.", file=sys.stderr)
        print("Start one with: ssh -fNM jupiter", file=sys.stderr)
        sys.exit(4)

    local_dest.mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-avz", remote_src, str(local_dest) + "/"]
    if dry_run:
        cmd.insert(2, "--dry-run")
        print(f"[dry-run] Would run: {' '.join(cmd)}")
    else:
        print(f"Rsyncing: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"rsync failed (exit {result.returncode}):\n{result.stderr}", file=sys.stderr)
        sys.exit(4)

    # Also discover and rsync descendant runs (for resumed training chains)
    if not dry_run:
        _rsync_jupiter_descendants(run_id)

    return JUPITER_MIRROR_ROOT


def _rsync_jupiter_descendants(root_id: str):
    """Discover and rsync the full descendant tree from Jupiter (single SSH call)."""
    discovery_script = (
        "import pathlib\n"
        f"base = pathlib.Path('{JUPITER_REMOTE_BASE}')\n"
        "parent_map = {}\n"
        "for d in base.iterdir():\n"
        "    tag_file = d / 'tags' / 'mlflow.parentRunId'\n"
        "    if tag_file.exists():\n"
        "        parent_map[d.name] = tag_file.read_text().strip()\n"
        f"queue = ['{root_id}']\n"
        "visited = set()\n"
        "while queue:\n"
        "    cur = queue.pop(0)\n"
        "    if cur in visited:\n"
        "        continue\n"
        "    visited.add(cur)\n"
        "    children = [rid for rid, pid in parent_map.items() if pid == cur]\n"
        "    for c in children:\n"
        "        print(c)\n"
        "    queue.extend(children)\n"
    )
    result = subprocess.run(
        ["ssh", JUPITER_SSH_HOST, "python3"],
        input=discovery_script, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"[warn] Jupiter descendant discovery failed (exit {result.returncode}); "
              f"continuing with local data only.", file=sys.stderr)
        return

    descendants = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not descendants:
        return

    print(f"Found {len(descendants)} descendant(s) of {root_id} on Jupiter")
    for desc_id in descendants:
        local_desc = JUPITER_MIRROR_ROOT / desc_id
        if local_desc.is_dir() and (local_desc / "metrics").exists():
            continue  # already mirrored
        remote = f"{JUPITER_SSH_HOST}:{JUPITER_REMOTE_BASE}/{desc_id}/"
        local_desc.mkdir(parents=True, exist_ok=True)
        print(f"  Rsyncing descendant: {desc_id}")
        subprocess.run(
            ["rsync", "-avz", remote, str(local_desc) + "/"],
            capture_output=True, text=True,
        )


def _ensure_jupiter_descendants_synced(run_id: str):
    """Check SSH availability and sync descendants if SSH is up."""
    check = subprocess.run(
        ["ssh", "-O", "check", JUPITER_SSH_HOST],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        return  # SSH not available; best-effort, use whatever is already mirrored
    _rsync_jupiter_descendants(run_id)


def locate_mlflow_experiment_dir(
    run_id: str, origin: str, *, local_only: bool = False, dry_run: bool = False,
) -> tuple[Path | None, str]:
    """Return (experiment_dir, discovery_note) or (None, reason)."""

    if origin == "atos":
        exp_dir = _find_local_mlflow_dir(run_id)
        if exp_dir:
            return exp_dir, f"Local ATOS: {exp_dir}/{run_id}"
        return None, f"MLflow run {run_id} not found in {ATOS_MLFLOW_ROOT}"

    if origin == "jupiter":
        # Check mirror first
        exp_dir = _find_jupiter_mirror(run_id)
        if exp_dir:
            # Root is mirrored; still ensure descendants are synced
            if not local_only:
                _ensure_jupiter_descendants_synced(run_id)
            return exp_dir, f"Jupiter mirror: {exp_dir}/{run_id}"
        # Also check ATOS (some Jupiter runs are synced to ATOS)
        exp_dir = _find_local_mlflow_dir(run_id)
        if exp_dir:
            # ATOS may have incomplete descendants; also sync from Jupiter
            if not local_only:
                _rsync_jupiter(run_id, dry_run=dry_run)
                mirror = _find_jupiter_mirror(run_id)
                if mirror:
                    return mirror, f"Jupiter mirror (synced, ATOS fallback): {mirror}/{run_id}"
            return exp_dir, f"Local ATOS (Jupiter-trained): {exp_dir}/{run_id}"
        if local_only:
            return None, "Jupiter MLflow logs not mirrored and --local-only set"
        # Rsync from Jupiter
        exp_dir = _rsync_jupiter(run_id, dry_run=dry_run)
        if dry_run:
            return None, f"[dry-run] Would rsync from Jupiter to {exp_dir}/{run_id}"
        return exp_dir, f"Rsynced from Jupiter: {exp_dir}/{run_id}"

    if origin == "leonardo":
        return None, "Leonardo MLflow import not yet supported"

    return None, f"Unknown training origin: {origin}"


# ---------------------------------------------------------------------------
# Phase C — load, plot, store
# ---------------------------------------------------------------------------


def _load_and_filter(experiment_dir: Path, run_id: str) -> dict:
    """Load only the requested run family, bypassing experiment-wide filtering."""
    from ._loader import load_run_family  # noqa: E402

    return load_run_family(experiment_dir, run_id)


def _generate_plots(runs: dict, run_root: Path):
    """Generate the standard plot bundle into run_root."""
    from ._plot import plot_key_vars, plot_overview, plot_all_vars  # noqa: E402

    plot_key_vars(runs, output=run_root / "key_vars.png")
    plot_overview(runs, output=run_root / "overview.png")
    try:
        plot_all_vars(runs, output=run_root / "all_vars.png")
    except ValueError:
        print("Skipping all_vars.png — no per-variable metrics found.")


def _store_metrics(runs: dict, run_root: Path, import_log_lines: list[str]):
    """Store raw metrics JSON and import log under data/mlflow/."""
    data_dir = run_root / "data" / "mlflow"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Convert metrics to JSON-serializable form
    metrics_out = {}
    for name, data in runs.items():
        metrics_out[name] = {
            "run_id": data["run_id"],
            "max_step": data["max_step"],
            "metrics": data["metrics"],
        }

    with open(data_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved: {data_dir / 'metrics.json'}")

    with open(data_dir / "import_log.txt", "w") as f:
        f.write("\n".join(import_log_lines) + "\n")
    print(f"Saved: {data_dir / 'import_log.txt'}")


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


def _already_generated(run_root: Path, run_id: str) -> bool:
    """Return True if plots already exist for this exact run_id."""
    metrics_json = run_root / "data" / "mlflow" / "metrics.json"
    if not metrics_json.exists():
        return False
    if not (run_root / "key_vars.png").exists():
        return False
    try:
        with open(metrics_json) as f:
            data = json.load(f)
        # Check any entry has matching run_id
        for entry in data.values():
            if entry.get("run_id") == run_id:
                return True
    except (json.JSONDecodeError, KeyError):
        pass
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Import MLflow training loss and generate plots for eval run roots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python mlflow_import.py --checkpoint-path ~/scratch/aifs/checkpoint/<UUID>/last.ckpt --run-root /tmp/test/
              python mlflow_import.py --checkpoint-path ~/scratch/aifs/checkpoint/<UUID>/last.ckpt --run-root /tmp/test/ --dry-run
        """),
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path,
                        help="Absolute path to the .ckpt file")
    parser.add_argument("--run-root", required=True, type=Path,
                        help="Eval run root directory for outputs")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip remote MLflow discovery (no rsync)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without executing")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if already generated")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()

    # ── validate checkpoint ──────────────────────────────────────────────
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # ── Phase A: discover ────────────────────────────────────────────────
    run_id = _extract_uuid(checkpoint_path)
    print(f"Checkpoint UUID: {run_id}")

    metadata = _read_checkpoint_metadata(checkpoint_path)
    origin = _detect_training_origin(metadata)
    print(f"Training origin: {origin}")

    import_log = [
        f"mlflow_import.py — {datetime.now(timezone.utc).isoformat()}",
        f"checkpoint: {checkpoint_path}",
        f"run_id: {run_id}",
        f"training_origin: {origin}",
    ]

    # ── idempotency check ────────────────────────────────────────────────
    if not args.force and _already_generated(run_root, run_id):
        print(f"Already generated for {run_id}. Use --force to regenerate.")
        sys.exit(0)

    # ── Phase B: locate MLflow logs ──────────────────────────────────────
    experiment_dir, discovery_note = locate_mlflow_experiment_dir(
        run_id, origin,
        local_only=args.local_only,
        dry_run=args.dry_run,
    )
    import_log.append(f"discovery: {discovery_note}")
    print(f"Discovery: {discovery_note}")

    if experiment_dir is None:
        print(f"MLflow logs not found: {discovery_note}", file=sys.stderr)
        import_log.append("result: not_available")
        # Still store the import log so the caller knows why
        if not args.dry_run:
            run_root.mkdir(parents=True, exist_ok=True)
            data_dir = run_root / "data" / "mlflow"
            data_dir.mkdir(parents=True, exist_ok=True)
            with open(data_dir / "import_log.txt", "w") as f:
                f.write("\n".join(import_log) + "\n")
        sys.exit(2)

    if args.dry_run:
        import_log.append("result: dry_run")
        print("[dry-run] Would load metrics, generate plots, and store data.")
        print("Import log would contain:")
        for line in import_log:
            print(f"  {line}")
        sys.exit(0)

    # ── Phase C: load, plot, store ───────────────────────────────────────
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from: {experiment_dir}")
    runs = _load_and_filter(experiment_dir, run_id)

    if not runs:
        print(f"No matching run found for {run_id} in {experiment_dir}", file=sys.stderr)
        import_log.append("result: no_matching_run")
        sys.exit(2)

    run_name = next(iter(runs))
    run_data = runs[run_name]
    import_log.append(f"run_name: {run_name}")
    import_log.append(f"max_step: {run_data['max_step']}")
    import_log.append(f"n_metrics: {len(run_data['metrics'])}")

    try:
        print(f"Generating plots in: {run_root}")
        _generate_plots(runs, run_root)
        import_log.append("result: success")
    except Exception as e:
        print(f"Plot generation failed: {e}", file=sys.stderr)
        import_log.append(f"result: plot_failure ({e})")
        _store_metrics(runs, run_root, import_log)
        sys.exit(3)

    _store_metrics(runs, run_root, import_log)
    print("Done.")


if __name__ == "__main__":
    main()
