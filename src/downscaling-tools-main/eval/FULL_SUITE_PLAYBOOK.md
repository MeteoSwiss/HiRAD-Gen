# Full Eval Suite Playbook

## Canonical: Pipeline Generation (Recommended)

Generate and submit a full SLURM dependency chain:

```bash
cd /home/ecm5702/dev/downscaling-tools
source /home/ecm5702/dev/.ds-dyn/bin/activate

python -m eval.jobs.pipeline \
    --lane o96_o320 --host atos_ac \
    --checkpoint /path/to/checkpoint \
    --output-dir /home/ecm5702/scratch/eval/my_run/pipeline/

# Review generated scripts, then submit:
bash /home/ecm5702/scratch/eval/my_run/pipeline/submit_pipeline.sh
```

This generates:
- `01_predict.sbatch` — prediction generation (GPU)
- `02_eval_tc.sbatch` — TC evaluation
- `02_eval_spectra.sbatch` — spectra computation
- `02_eval_surface.sbatch` — surface nMSE
- `02_eval_region_plot.sbatch` — regional comparison plots
- `03_scoreboard.sbatch` — aggregate scores
- `submit_pipeline.sh` — SLURM dependency chain launcher

All scripts call `python -m eval.cli` with correct arguments.

## Direct CLI (Interactive)

For manual evaluation on existing predictions:

```bash
python -m eval.cli evaluate \
    --predictions-dir /path/to/predictions/ \
    --lane o96_o320 \
    --checkpoint /path/to/checkpoint

python -m eval.cli scoreboard \
    --eval-dir /path/to/eval/output/ \
    --lane o96_o320
```

## Legacy: One-Command Shell Script

> **Note:** The legacy launchers (`launch_full_eval_suite.sh`, `codex_eval`) still work
> but are deprecated in favor of `eval.jobs.pipeline` which provides better resource
> control, evaluator-level parallelism, and consistent `eval.cli` invocations.

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh --expver <EXPVER>
```

## Useful Monitoring
```bash
squeue -u $(whoami) -n eval-o96_o320-*
sacct -j <comma-separated-jobids> --format=JobID,JobName%24,State,Elapsed,ExitCode
```
