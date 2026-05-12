> **Status: legacy (pre-refactor).** This playbook is preserved under
> `eval/jobs/legacy_docs/` for historical reference. For canonical
> post-refactor workflows see `eval/jobs/README.md`,
> `eval/jobs/templates/README.md`, and `eval/FULL_SUITE_PLAYBOOK.md`.

# Eval Agent Playbook (Parallel Bundle)

Use this as the default when a user asks to evaluate a run.

## Required Behavior

- Launch independent eval components in parallel.
- Keep local plotting to one representative date (5 lead times) unless the user asks for all dates.
- Use short quaver windows (5 days) by default.
- Every `tc_normed_pdfs*.pdf` generation now also writes a sibling `*.stats.json` file in the same output directory.
- For Idalia, `*.stats.json` now includes an `extreme_tail` section with default thresholds:
  - `MSLP in [980, 990] hPa`
  - `10m wind > 25 m/s`
  - plus an `extreme_score` ranking that combines both tails.
- For all TC requests, agents must explicitly report these extreme-tail values for the target run:
  - `extreme_score`, `mslp_980_990_fraction`, `wind_gt_25_fraction`
  - plus the thresholds used in that file.
- For all TC requests, agents must also compare the target run against all previously evaluated experiments currently available in `/home/ecm5702/scratch/eval` by scanning:
  - `tc_extreme_tail_*.json`
  - `tc_normed_pdfs_*from_predictions.stats.json`
  - and return a ranked cross-experiment table (highest `extreme_score` first).
- For all "extreme" requests, this comparison table is mandatory and must include the target run plus all other analyzed experiments discoverable in `/home/ecm5702/scratch/eval`.
- If any analyzed experiment is missing extreme stats, agents must recalculate the missing stats first, then regenerate the ranked comparison table before final reporting.
- Final response must always include:
  - the absolute path to the generated comparison table file,
  - the absolute output directory path(s),
  - experiment config used for the target run (checkpoint/run_id, sampling params, script/command, job id(s)).
- Record all submitted jobs and monitor commands in the active epic task note under `/nfs/dh2_home_a/ecm5702/dev/docs/epics/checkpoint-eval-pipeline/in-progress/`.

## A) O320 -> O1280 Manual Inference + Eval Bundle

Use the canonical helper below as the default weak-agent route for fresh `o320 -> o1280` manual-inference work:

```bash
bash /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
```

Edit only the `USER SETTINGS` block first. The helper will:
- validate the checkpoint profile and ensure the lane is `o320_o1280`
- auto-resolve the stack flavor from the checkpoint
- default to `PHASE=proxy` unless the operator explicitly selects `continue-full` or `full-only`
- validate the raw source GRIB tree and rebuild strict truth-aware bundles into `<RUN_ROOT>/bundles_with_y`
- force the tested O1280 predict posture (`4` GPUs, `32` CPUs, `24h`)
- render host-safe dated submit copies under `/home/ecm5702/dev/jobscripts/submit/<YYYYMMDD>/`
- submit, with `afterok` dependencies:
  - `build_o320_o1280_truth_bundles.sbatch`
  - `strict_manual_predict_x_bundle.sbatch`
  - `local_plots_one_date_from_predictions.sbatch`
  - `spectra_proxy_from_predictions.sbatch` on AG or `spectra_ecmwf_from_predictions.sbatch` on AC
  - `tc_eval_from_predictions.sbatch` with `native` on AG or `regridded` on AC

Default commands:

```bash
CHECKPOINT_PATH=<CKPT_PATH> PHASE=proxy \
  bash /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh

CHECKPOINT_PATH=<CKPT_PATH> PHASE=continue-full RUN_ID_OVERRIDE=<same_run_id> \
  bash /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
```

For proxy-like reduced scopes, set:
- `BUNDLE_PAIRS=YYYYMMDD:HH,...`
- optionally `LOCAL_PLOT_DATE`
- optionally `LOCAL_PLOT_EXPECTED_COUNT=auto`

Do **not** use `/home/ecm5702/hpcperm/data/input_data/o320_o1280/idalia` as the strict prediction input root. That tree is the raw source GRIB root; strict prediction must read from rebuilt `bundles_with_y`.

Do **not** fork ad hoc `/tmp/*.sbatch` launchers for this lane.

## B) Predictions Run (`run-id`) - Eval Only

Assumes predictions already exist at:
`/home/ecm5702/scratch/eval/<RUN_ID>/predictions/predictions_*.nc`

1) One-date local plots:

```bash
cp \
  /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/local_plots_one_date_from_predictions.sbatch \
  /home/ecm5702/dev/jobscripts/local_plots_one_date_<RUN_ID>.sbatch
# Edit RUN_ROOT, RUN_ID, DATE, and EXPECTED_COUNT in the copied file, then:
sbatch /home/ecm5702/dev/jobscripts/local_plots_one_date_<RUN_ID>.sbatch
```

This writes canonical non-TC local plots under:
- `/home/ecm5702/scratch/eval/<RUN_ID>/local_plots_one_date/predictions_20230826_step024/`
- ...
- `/home/ecm5702/scratch/eval/<RUN_ID>/local_plots_one_date/predictions_20230826_step120/`

For new outputs, prefer the baseline filenames emitted by the canonical helper, for example:
- `amazon_forest_member01_baseline.pdf`
- `amazon_forest_member01_baseline.png`

2) TC plots:

```bash
cp \
  /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/tc_eval_from_predictions.sbatch \
  /home/ecm5702/dev/jobscripts/tc_eval_<RUN_ID>.sbatch
# Edit RUN_ROOT and RUN_ID in the copied file, then:
bash /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_tc_eval_from_predictions.sh \
  /home/ecm5702/dev/jobscripts/tc_eval_<RUN_ID>.sbatch
```

3) Spectra generation:

```bash
cp \
  /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/spectra_proxy_from_predictions.sbatch \
  /home/ecm5702/dev/jobscripts/spectra_proxy_<RUN_ID>.sbatch
# Edit RUN_ROOT, RUN_ID, and filter settings in the copied file, then:
sbatch /home/ecm5702/dev/jobscripts/spectra_proxy_<RUN_ID>.sbatch
```

On AC, when trusted ECMWF spectra are explicitly needed, switch to:

```bash
cp \
  /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/spectra_ecmwf_from_predictions.sbatch \
  /home/ecm5702/dev/jobscripts/spectra_ecmwf_<RUN_ID>.sbatch
sbatch /home/ecm5702/dev/jobscripts/spectra_ecmwf_<RUN_ID>.sbatch
```

4) Run manifest:

Prediction-first routes should already have `/home/ecm5702/scratch/eval/<RUN_ID>/EXPERIMENT_CONFIG.yaml` from the strict manual-inference template. If it is missing, treat that as a workflow bug and backfill it immediately rather than inventing a second manifest surface.

## C) Expver Run (Full Eval Family)

Use existing launcher and let it submit components in parallel:

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh --expver <EXPVER>
```

Recommended short-window flags:

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh \
  --expver <EXPVER> \
  --quaver-first-date 20230826 \
  --quaver-last-date 20230830 \
  --spectra-date 20230826/to/20230830/by/1
```

## Monitoring

```bash
squeue -j <JOB1>,<JOB2>,<JOB3>
sacct -j <JOB1>,<JOB2>,<JOB3> --format=JobID,JobName%30,QOS,State,ExitCode,Elapsed,Timelimit,NodeList,Reason -n -P
```

## TC Extreme Comparison (Required Reporting)

Use this when summarizing TC results so users always get cross-run extreme-tail context:

```bash
find /home/ecm5702/scratch/eval -type f \( -name 'tc_extreme_tail_*.json' -o -name 'tc_normed_pdfs_*from_predictions.stats.json' \) | sort
```

Extract/merge rows and rank by `extreme_score` descending. Include per-row:
- `exp`
- `extreme_score`
- `mslp_980_990_fraction`
- `wind_gt_25_fraction`
- thresholds used for each row
- source file path

Write the ranked table to a stable artifact path (for example `/home/ecm5702/scratch/eval/tc_extreme_scoreboard_all_exps.tsv` and/or `<RUN_DIR>/tc_extreme_scoreboard_all_exps.tsv`) and share that path.

## Clean Readable Scoreboards (VS Code Friendly)

Regenerate the two markdown scoreboards:

```bash
python /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/generate_clean_scoreboards.py
```

Outputs:
- `/home/ecm5702/scratch/eval/scoreboards/prepml_all_tc_reproduction_scoreboard.md`
- `/home/ecm5702/scratch/eval/scoreboards/global_extreme_scoreboard.md`

## Notes

- MARS/FDB retrieval requests can fail transiently or produce incomplete staging. If a TC retrieve job fails or expected files are missing, restart the retrieve step and then rerun dependent TC plotting jobs.
- Before launching TC plots, verify staged inputs exist under `/home/ecm5702/perm/reference/o96_o320/tc/<event>/surface_pf_ENFO_O320_<EXPVER>_YYYYMMDD.grib`.
- If a previous all-dates eval job is running and user requests one-date-only local plots, cancel the broader job and replace it with one-date eval-only submission.
- For checkpoint mode (`python -m eval.run checkpoint ...`), keep sigma evaluator enabled unless the user explicitly asks to skip.
- Do not use `python -m eval.tc.workflows pdf --ml-expids ENFO_O320_0001` as the ML curve for run-id evaluations; this duplicates the ENFO reference and is not run-specific.
