# Strict Manual-Inference And Eval Templates

> **Canonical entry point:** `python -m eval.cli run --checkpoint <CKPT_PATH> --lane <LANE> --host <HOST>`
> Archived (covered by `eval.cli`): old shell flow scripts, `scoreboard_*_step.sbatch`, `scoreboard_write_from_predictions.sbatch`, `tc_eval_from_predictions.sbatch`, `spectra_proxy_from_predictions.sbatch`, `surface_loss_from_predictions.sbatch`, `regional_suite_from_predictions.sbatch`, `local_plots_one_date_from_predictions.sbatch`, `build_o*_truth_bundles.sbatch` (superseded by `eval.cli prepare` / `eval.cli run --source-grib-root`), all submit helpers, and their Python helpers — all moved to `archive/`.

This directory is the canonical home for repo-specific eval and manual-inference templates used by `downscaling-tools`.
If copies exist under `jobscripts/`, treat them as mirrors, not the source of truth.
This index lists only the templates that are actually present in this directory.

## Surface split
- Canonical maintained templates live here.
- Rendered disposable submit-ready artifacts belong under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/`.
- The older mirror at `/home/ecm5702/dev/jobscripts/templates/codex_login_node_templates/` is a compatibility surface, not the source of truth for new shared edits.
- For policy and routing, also read `/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/docs/docs/instructions/inference-launching.md`.

**Before submitting any job**, source the preflight script or run it standalone:
```bash
# Standalone check
bash eval/jobs/templates/preflight_eval_check.sh --predictions-dir /path --stack new --job-type predict75

# Or source from within a template
source preflight_eval_check.sh
preflight_cluster
preflight_venv "new"
preflight_predictions_dir "/path/to/predictions" 25
preflight_summary
```

## Template Matrix

### Inference
- `predict_recovery.sbatch`
  - **Recovery for walltime-killed prediction runs.**
  - Auto-detects missing prediction files and relaunches only remaining date/step combos.
  - Targets an existing run directory and reruns only the missing files.
  - Works for `new` and `old` stack, AC and AG.

### Preflight
- `preflight_eval_check.sh`
  - **Source this before any submission** to validate cluster, venv, data, QOS, and walltime.
  - Also works standalone: `bash preflight_eval_check.sh --help`.

### Training
- `train_o48_o96.sbatch`
  - Canonical `o48 -> o96` training launcher.
  - The `o48_o96` top-level config uses `data: downscaling_o48` which carries
    the correct 11-var forcing list and normalizer. The global `training/default.yaml`
    `lr.min` is `1e-7` (safe floor). This template only adds run-specific overrides
    (sigma schedule, max_steps, batch size, PL variable group fix).
  - Edit the USER SETTINGS block and copy to
    `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/` before submitting.

## Canonical Upstream Tools

These are the authoritative eval tools in `downscaling-tools`. Templates wrap them;
do not reimplement their logic in ad-hoc scratch scripts.

| Tool | Path | Purpose |
|------|------|---------|
| TC data request | `eval/tc/all_events_request.sh` | MARS request for TC reference GRIBs (edit EXPID) |
| Spectra pipeline | `eval/spectra/grb_to_spectra.sh` | Full MARS→gptosp→compute spectra pipeline |
| ECMWF spectra evaluator | `eval/evaluators/spectra_ecmwf/` | `eval.cli evaluate --only spectra_ecmwf` (AC-only) |
| MLflow evaluator | `eval/evaluators/mlflow/` | `eval.cli evaluate --only mlflow` (requires `--checkpoint`) |

## Design Invariants
- Manual-inference templates keep host/env resolution explicit and validated:
  - `ac + new -> /home/ecm5702/dev/.ds-dyn/bin/activate`
  - `ag + new -> /home/ecm5702/dev/.ds-ag/bin/activate`
  - `ac + old -> /home/ecm5702/dev/.ds-old/bin/activate`
  - `ag + old -> /home/ecm5702/dev/.ds-ag-old/bin/activate`
- Eval templates detect the cluster automatically and activate the correct env.
- Output paths stay explicit:
  - users set `RUN_ROOT`, `RUN_ID`, and optional output tags directly
  - templates write into the chosen run tree, not into hidden temp locations

## Resource Baselines (Empirical)

These are based on observed job outcomes from the checkpoint-eval-pipeline epic.

| Job Type | Walltime | Memory | QOS | Notes |
|----------|----------|--------|-----|-------|
| predict25 (O320, GPU) | 12:00:00 | default | ng | 25 files, single GPU |
| predict75 (O320, GPU) | 48:00:00 | default | ng | 61/75 in 12h observed; use 48h |
| predict25 (O1280, GPU) | 24:00:00 | default | ng | ~2-3x slower than O320 |
| Spectra ECMWF (AC) | 48:00:00 | 128G | nf | 300 gptosp transforms; resumable |

## Cluster / QOS Rules

| Cluster | Available QOS | Notes |
|---------|---------------|-------|
| AC | `nf` (CPU), `np`, `ef`, `ng` (GPU) | Use `nf` for CPU eval jobs; the TC submit helper defaults to this on AC |
| AG | `ng` only | Use `--gpus-per-node=0` for CPU-only; `nf` does NOT work on AG |

Common mistakes to avoid:
- Requesting `qos=nf` on AG → immediate rejection.
- Requesting `qos=ng` + `--gpus-per-node=0` on AC → can fail with `QOSMinGRES`; prefer `qos=nf` with no GPU request.
- Omitting `--gpus-per-node=0` on AG → Slurm auto-adds a GPU.
- Requesting `>128G` on AC `nf` → may hit `QOSMaxMemoryPerJob`.

## How To Use
1. Copy a template from this folder.
2. Edit only the `USER SETTINGS` block.
3. Optionally run `bash preflight_eval_check.sh` to validate before submitting.
4. Submit with `sbatch <template>.sbatch`.

Do not edit rendered copies under `/home/ecm5702/dev/jobscripts/submit/` when the goal is to change shared template behavior for future runs.

## Smooth Routes By Goal
- `o48 -> o96` training with correct forcing/LR overrides:
  - copy `train_o48_o96.sbatch`, edit USER SETTINGS, `sbatch`
- `o48 -> o96` rebuild strict Humberto bundles only:
  - `python -m eval.cli prepare --lane o48_o96 --source-grib-root <GRIB_ROOT> --bundle-dir <BUNDLE_DIR>`
- `o48 -> o96` full eval with truth-aware bundles:
  - `python -m eval.cli run --checkpoint <CKPT_PATH> --lane o48_o96 --source-grib-root <GRIB_ROOT>`
- `o48 -> o96` full eval from pre-built bundles:
  - `python -m eval.cli run --checkpoint <CKPT_PATH> --lane o48_o96 --host atos_ac`
- `o96 -> o320` full eval:
  - `python -m eval.cli run --checkpoint <CKPT_PATH> --lane o96_o320 --host atos_ac`
- `o320 -> o1280` full eval with truth-aware bundles:
  - `python -m eval.cli run --checkpoint <CKPT_PATH> --lane o320_o1280 --source-grib-root <GRIB_ROOT>`
- `o1280 -> o2560` full eval with truth-aware bundles:
  - `python -m eval.cli run --checkpoint <CKPT_PATH> --lane o1280_o2560 --source-grib-root <GRIB_ROOT>`
- Re-run a single failed pillar from existing predictions:
  - `python -m eval.cli evaluate --predictions-dir <RUN_ROOT>/data/predictions --lane <LANE> --only <pillar>`
- Recovery:
  - edit `predict_recovery.sbatch`
- ECMWF spectra (AC only) on existing predictions:
  - `python -m eval.cli evaluate --predictions-dir <DIR> --lane <LANE> --only spectra_ecmwf`
- MLflow training-loss plots:
  - `python -m eval.cli evaluate --predictions-dir <DIR> --lane <LANE> --only mlflow --checkpoint <CKPT>`

## Notes
- Resolver tests for the strict inference templates live in:
  - `/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/tests/test_checkpoint_profile.py`

## Canonical Entry Points Used By Templates

Shared templates should prefer `python -m eval.cli` for the unified evaluation framework, `python -m eval.predict.main` for repo-native prediction generation, and `python -m manual_inference.prediction.predict` for one-bundle manual inference. Rendered historical scripts under `/home/ecm5702/dev/jobscripts/submit/` are run artifacts and should not be rewritten for documentation-only migrations.
