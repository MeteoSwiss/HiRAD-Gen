# Architecture

Target architecture for the evaluation framework. Where the current codebase
diverges from this target, a **Current state** annotation marks the gap. The
codebase evolves to match this document.

## 1. Layered Overview

```
Input sources
  |-- checkpoint (research)  -> eval.predict -> predictions_*.nc
  |-- MARS expver (prepml)   -> eval.run mars-expver -> predictions + region plots
                                        |
                              predictions directory
                                        |
                              eval.cli evaluate
                                        |
                    eval.evaluators.<name>.run()    (computation)
                    eval.evaluators.<name>.score()  (metrics extraction)
                    eval.evaluators.<name>.plot()   (visualization)
                                        |
                    eval.scoreboard.aggregator      (collect metrics.json)
                    eval.scoreboard.formatter       (CSV / markdown)
                                        |
                    eval.jobs.pipeline / renderer    (HPC orchestration)
```

Supporting layers:
- `eval/config/` -- lane, host, and event YAML configuration
- `eval/discovery/` -- prediction file finding and checkpoint identification
- `eval/shared/` -- common grid and plotting utilities
- `eval/paths.py` -- canonical path resolution

**Current state**: The prepml path (`eval.run mars-expver`) is the legacy CLI,
still used by the full-suite launcher. The checkpoint path flows through
`eval.cli`. Both produce a predictions directory that the evaluator framework
consumes.

## 2. Evaluator Architecture

Each evaluator is a self-contained package under `eval/evaluators/<name>/`:

```
eval/evaluators/tc/
|-- __init__.py       # Exports: run(), score(), plot(); EVALUATOR_SPEC
|-- runner.py         # Orchestration -- calls kernel functions
|-- scorer.py         # Metrics extraction -- calls kernel scoring math
|-- plotter.py        # Visualization
|-- kernel/           # Domain logic (data loading, statistics, grid ops)
    |-- workflows.py
    |-- stats.py
    |-- data_types.py
    |-- events.py
    |-- grid.py
    |-- loading_grib.py
    |-- loading_predictions.py
    |-- member_plot.py
    |-- pdf_plot.py
    |-- plot_config.py
```

The same pattern applies to all evaluators:

| Evaluator | Kernel contents |
|---|---|
| `tc` | TC workflows, stats, loading, plotting |
| `spectra` | Spectral harmonics computation, comparison plots |
| `surface` | Surface nMSE scoring math, normalization (self-contained; no separate legacy runner) |
| `region_plot` | Six-panel region plotting, coordinate/variable utils |
| `sigma` | Noise schedule sweeps, sigma evaluator logic |
| `mechanistic` | Weight diagnostics / interpretability |
| `intermediate` | Intermediate diffusion step visualization |

`eval/scoreboard/` contains only the canonical aggregation layer
(`aggregator.py`, `formatter.py`, `types.py`). Per-domain scoring math lives
inside the respective evaluator's `kernel/` or `scorer.py`.

### Evaluator Convention

Each `__init__.py` exports up to three functions:

```python
def run(predictions_dir, lane_config, eval_config) -> Path:
def score(results_dir, lane_config, eval_config) -> list[dict]:
def plot(results_dir, lane_config, eval_config, output_dir) -> list[Path]:

EVALUATOR_SPEC = {
    "name": "tc",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
```

Not every evaluator needs all three. This is convention, not a base class.

### Evaluator Rules

- Each evaluator writes data under `data/<name>/` and plots under `plots/<name>/`.
- No evaluator writes outside its own subdirectories.
- No cross-evaluator imports.
- Evaluators import from `eval.config`, `eval.discovery`, `eval.shared`, their
  own `kernel/`, and stdlib. Never from `eval.jobs` or another evaluator.

**Current state**: Kernels currently live in separate top-level directories
(`eval/tc/`, `eval/spectra/`, `eval/sigma_evaluator/`, etc.) and scoring math
lives in `eval/scoreboard/{tc,spectra,surface}.py`. These will be consolidated
into their evaluator packages as sbatch templates migrate to `eval.cli`.

## 3. Configuration

Lane, host, and event configuration lives in `eval/config/` as YAML:

```
eval/config/
|-- lanes/          # One file per resolution lane
|   |-- o48_o96.yaml
|   |-- o96_o320.yaml
|   |-- o320_o1280.yaml
|   |-- o1280_o2560.yaml
|-- hosts/          # One file per HPC host
|   |-- atos_ac.yaml
|   |-- atos_ag.yaml
|-- events/         # TC event definitions
|   |-- idalia.yaml
|   |-- franklin.yaml
|   |-- ...
|-- loader.py       # Reads YAML, validates required keys, returns dict
```

**Lane YAML** is the central config file. It contains: predict defaults (dates,
steps, members), per-evaluator parameters, evaluator groups, region definitions,
and reference data paths.

**Evaluator groups** control which evaluators run by default:

```yaml
evaluator_groups:
  default: [tc, spectra, surface, region_plot]
  diagnostics: [sigma, mechanistic, intermediate]
  experimental: [quaver]
```

`eval.cli evaluate` runs the `default` group unless `--only` overrides.
`--include-diagnostics` adds the diagnostics group.

**Config precedence**: CLI args > lane YAML > host YAML defaults. Every run emits
`effective_config.json` recording the resolved snapshot.

**`eval_config` boundary**: `eval_config` is `lane_config[evaluator_name]` -- the
evaluator-specific subsection. Evaluators read evaluator-specific values only from
`eval_config` and cross-cutting values only from `lane_config`.

## 4. Output Directory Contract

Every evaluation run produces a structured output directory with a clear
data/plots separation:

```
<scratch_eval_root>/<lane>/<run_id>/
|-- effective_config.json
|-- data/
|   |-- predictions/            # Prediction NetCDFs
|   |-- tc/                     # TC stats, per-event results
|   |   |-- metrics.json
|   |-- spectra/                # Spectral amplitudes, per-variable results
|   |   |-- metrics.json
|   |-- surface/                # Surface loss results
|   |   |-- metrics.json
|   |-- sigma/                  # Sigma sweep results
|   |   |-- metrics.json
|   |-- scoreboard/
|       |-- scores.csv
|       |-- scores.md
|-- plots/
|   |-- tc/                     # TC PDFs, member field maps
|   |-- spectra/                # Spectra comparison plots
|   |-- region_plot/            # Six-panel regional comparisons
|   |-- sigma/                  # Sigma sweep plots
|   |-- mechanistic/            # Weight diagnostic plots
|   |-- intermediate/           # Intermediate diffusion step plots
|-- logs/
```

**Rules:**
- Each evaluator writes data under `data/<name>/` and plots under
  `plots/<name>/`. No evaluator writes outside its own subdirectories.
- Scored evaluators produce `data/<name>/metrics.json` -- a list of
  `{"metric": str, "value": float, "unit": str}` records.
- `run()` refuses to overwrite existing output unless `overwrite=True` is passed.
- `effective_config.json` is emitted twice: after config resolution (before
  expensive work) and updated at completion with status and actual outputs.

**Current state**: Evaluators currently write data and plots together under
`evaluators/<name>/`. The data/plots separation is a target restructuring.

## 5. CLI

Single entry point: `python -m eval.cli <subcommand>`. Never bare `eval`.

```bash
# Full pipeline: predict + evaluate + scoreboard
python -m eval.cli run --checkpoint <path> --lane o96_o320 [--host atos_ac] [--only tc,spectra]

# Predictions only
python -m eval.cli predict --checkpoint <path> --lane o96_o320

# Evaluate existing predictions
python -m eval.cli evaluate --predictions-dir <dir> --lane o96_o320 [--only tc,spectra,surface]

# Include diagnostics group
python -m eval.cli evaluate --predictions-dir <dir> --lane o96_o320 --include-diagnostics

# Scoreboard from existing evaluation results
python -m eval.cli scoreboard --eval-dir <dir> --lane o96_o320

# Dry run (print resolved config, don't execute)
python -m eval.cli run --checkpoint <path> --lane o96_o320 --dry-run
```

**Evaluator selection** follows three-step resolution:
1. `--only tc,spectra` -- run exactly those evaluators
2. `--include-diagnostics` -- default + diagnostics groups from lane YAML
3. Neither -- default group only

**Overrides**: `--members`, `--steps`, `--dates` override lane YAML predict
defaults. CLI always wins over YAML.

**Current state**: `eval.cli` is operational for all four subcommands. The prepml
path remains on the legacy CLI (`python -m eval.run mars-expver`) used by
`launch_full_eval_suite.sh`. Production sbatch templates have not migrated to
`eval.cli` yet -- they still call per-pillar legacy entry points directly.

## 6. HPC Job Orchestration

HPC submission is handled by two layers:

**Pipeline renderer** (`eval/jobs/pipeline.py`) generates a chain of sbatch
scripts with SLURM dependency linking (`--dependency=afterok:<jobid>`). A typical
chain:

```
predict.sbatch -> tc_eval.sbatch -> spectra_eval.sbatch -> surface_eval.sbatch -> scoreboard.sbatch
                                                                                       ^
                                                                                afterok on all eval jobs
```

**Template renderer** (`eval/jobs/renderer.py`) patches individual sbatch
templates with run-specific directives (QOS, partition, resource requests,
environment setup). Host YAML provides the environment:

```yaml
environment_setup:
  module_loads: ["ecmwf-toolbox", "python3/3.11"]
  exports: {"OMP_NUM_THREADS": "1"}
  venv_activate: "/path/to/venv/bin/activate"
```

**Resource profiles** (`eval/jobs/resources.py`) define per-evaluator HPC resource
requirements (nodes, GPUs, walltime, memory) so the pipeline can size each stage
correctly.

**Entry points:**
```bash
# Render + submit a full pipeline
python -m eval.jobs.pipeline --lane o96_o320 --host atos_ac --checkpoint <path>

# Render a single sbatch (dry run)
python -m eval.jobs.renderer --lane o96_o320 --host atos_ac --checkpoint <path> --dry-run

# One-command full eval suite (prepml/MARS expver path)
eval/jobs/launch_full_eval_suite.sh --expver <expver>
```

**Current state**: `pipeline.py` and `renderer.py` exist but are never invoked by
production workflows. The five `submit_*_flow.sh` scripts each re-implement their
own renderer and dependency chaining in bash + inline Python. The completeness
plan consolidates these into `pipeline.py`. `launch_full_eval_suite.sh` remains
the prepml front door and uses the legacy CLI internally.

## 7. Naming Conventions and Contracts

**Lane names**: `o48_o96`, `o96_o320`, `o320_o1280`, `o1280_o2560`
(underscore-separated, lowercase, `<input>_<output>` resolution).

**Host names**: `atos_ac`, `atos_ag` (underscore-separated, lowercase).

**Evaluator names**: `tc`, `spectra`, `surface`, `region_plot`, `sigma`,
`mechanistic`, `intermediate`. Must match directory name under
`eval/evaluators/`.

**Prediction files**: `predictions_YYYYMMDD_stepNNN.nc` (regex at
`eval/discovery/predictions.py`).

**Score record**: `{"metric": str, "value": float, "unit": str}` -- the handoff
format between evaluator scorers and the scoreboard aggregator.

**EVALUATOR_SPEC**: every evaluator's `__init__.py` exports this dict:
```python
EVALUATOR_SPEC = {
    "name": "tc",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
```

**Config paths**: all paths in YAML config files are absolute.

**Output root**: `<scratch_eval_root>/<lane>/<run_id>/`

**Import rules**:
- Evaluators import from `eval.config`, `eval.discovery`, `eval.shared`, their
  own `kernel/`, and stdlib.
- No cross-evaluator imports.
- No evaluator imports from `eval.jobs`.
