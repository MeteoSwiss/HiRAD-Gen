# downscaling-tools

ML-based atmospheric downscaling: training, inference, and evaluation tools for
ECMWF's diffusion-based downscaling pipeline.

## Canonical CLI

| Operation | Command |
|-----------|---------|
| Full eval pipeline | `python -m eval.cli run --lane <lane> --checkpoint <path> --host <host>` |
| Evaluate predictions | `python -m eval.cli evaluate --predictions-dir <dir> --lane <lane>` |
| Scoreboard | `python -m eval.cli scoreboard --eval-dir <dir> --lane <lane>` |
| Generate sbatch chain | `python -m eval.jobs.pipeline --lane <lane> --host <host> --checkpoint <path> --output-dir <dir>` |
| Render single sbatch | `python -m eval.jobs.renderer --lane <lane> --host <host> --checkpoint <path> --mode <mode>` |

## Subsystem Index

| Package | Purpose | Docs |
|---------|---------|------|
| `eval/` | Evaluation framework | [`eval/README.md`](eval/README.md) |
| `eval/predict/` | Prediction generation | [`eval/predict/README.md`](eval/predict/README.md) |
| `eval/evaluators/` | Evaluator modules (tc, spectra, surface, sigma, region_plot) | See eval/README.md |
| `eval/config/` | Lane/host YAML configuration | `eval/config/lanes/`, `eval/config/hosts/` |
| `eval/jobs/` | HPC job scripts + pipeline renderer | [`eval/jobs/README.md`](eval/jobs/README.md) |
| `manual_inference/` | Legacy inference scripts | [`manual_inference/README.md`](manual_inference/README.md) |

## Agent Context

Agent routing: `/home/ecm5702/dev/docs/AGENTS.md`
