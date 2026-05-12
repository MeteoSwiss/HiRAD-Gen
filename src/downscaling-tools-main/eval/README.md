# Eval

Evaluation utilities that consume `predictions_YYYYMMDD_stepNNN.nc` files produced
by `eval.predict.main` or by the unified CLI.

## Canonical CLI: `eval.cli`

The unified CLI is the **required** interface for all evaluation operations:

```bash
python -m eval.cli <subcommand> [args]
```

### Subcommands

**Full pipeline** (predict + evaluate + scoreboard):
```bash
python -m eval.cli run \
    --lane o96_o320 \
    --checkpoint /path/to/checkpoint \
    --host atos_ac
```

**Evaluate existing predictions**:
```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 \
    --checkpoint /path/to/checkpoint
```

**Run a single evaluator**:
```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 --only surface
```

**Generate scoreboard from completed evaluation**:
```bash
python -m eval.cli scoreboard \
    --eval-dir /path/to/eval/output/ \
    --lane o96_o320
```

Use `--dry-run` on any subcommand to print the resolved config as JSON.

Use `--include-diagnostics` to run the diagnostics group (sigma, mechanistic, intermediate) in addition to defaults.

## Pipeline Generation

Generate HPC sbatch chains with SLURM dependency chaining:

```bash
python -m eval.jobs.pipeline \
    --lane o96_o320 --host atos_ac \
    --checkpoint /path/to/checkpoint \
    --output-dir /path/to/pipeline/scripts/
```

This produces `01_predict.sbatch`, `02_eval_*.sbatch`, `03_scoreboard.sbatch`, and a `submit_pipeline.sh` launcher with `--dependency=afterok` chaining.

## Evaluator Architecture

Evaluators live in `eval/evaluators/<name>/` and follow a wrapper pattern:
- **tc, surface**: Native Python implementations wrapping legacy kernels
- **spectra, sigma, region_plot**: Subprocess wrappers around legacy modules
- **mechanistic, intermediate**: Stubs (not yet implemented)

Each evaluator exports `EVALUATOR_SPEC`, `run()`, `score()`, and optionally `plot()`.

Lane configuration: `eval/config/lanes/<lane>.yaml`
Host configuration: `eval/config/hosts/<host>.yaml`

## Backends (`eval/_backends/`)

Internal implementation details of the evaluator wrappers. **Never invoke directly.**

Contains: `tc/`, `spectra/`, `region_plotting/`, `sigma_evaluator/`, `weight_diagnostics/`,
`plot_intermediate/`, `quaver/`, and `scoreboard/{tc,spectra,surface,_surface_compute,_utils,canonical_data,row_matching}.py`.

These modules were moved here from their original top-level `eval/` locations as part of the
legacy quarantine. All imports have been updated. Old import paths (`eval.tc.*`, `eval.spectra.*`,
etc.) will fail immediately — this is intentional.

## Notebooks
- `eval/notebooks/00_eval_overview.ipynb`
- `eval/notebooks/01_unified_runner.ipynb`
- `eval/notebooks/02_intermediate_plots.ipynb`
- `eval/notebooks/03_region_plotting.ipynb`
- `eval/notebooks/04_sigma_evaluator.ipynb`
- `eval/notebooks/05_quaver.ipynb`
- `eval/notebooks/06_spectra.ipynb`
- `eval/notebooks/07_tc.ipynb`

## Prediction Generation

The `eval/predict/` package provides modular prediction generation from input bundles,
replacing the monolithic `generate_predictions_25_files.py`:

```bash
python -m eval.predict.main \
  --input-root /path/to/bundles \
  --out-dir /path/to/output \
  --name-ckpt /path/to/checkpoint.ckpt \
  --dates 20230826,20230827,20230828,20230829,20230830 \
  --steps 24,48,72,96,120 \
  --members 1,2,3,4,5,6,7,8,9,10
```

See [`eval/predict/README.md`](predict/README.md) for full documentation.

## Archive (`eval/archive/`)

Contains retired scripts and old templates. Not used in live workflows.
