from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from eval.jobs import slurm_jobs
from eval.paths import DEFAULT_EVAL_ROOT

TERMINAL_OK = slurm_jobs.TERMINAL_OK
TERMINAL_BAD = slurm_jobs.TERMINAL_BAD
TERMINAL_ALL = slurm_jobs.TERMINAL_ALL
JobTrack = slurm_jobs.JobTrack


def _run(cmd: list[str]) -> str:
    return slurm_jobs.run_checked(cmd)


def _submit(script: Path, dependency: str | None = None) -> str:
    return slurm_jobs.submit(_run, script, dependency)


def _cancel(job_id: str) -> None:
    slurm_jobs.cancel(job_id)


def _job_state(job_id: str) -> str:
    return slurm_jobs.job_state(_run, job_id)


def _write_state(state_file: Path, jobs: dict[str, JobTrack], expver: str) -> None:
    payload = {
        "expver": expver,
        "updated_epoch": slurm_jobs.updated_epoch(),
        "jobs": slurm_jobs.jobs_payload(jobs),
    }
    slurm_jobs.write_json(state_file, payload)


def _all_terminal(jobs: dict[str, JobTrack]) -> bool:
    return all(j.state in TERMINAL_ALL for j in jobs.values())


def _regenerate_scripts(launch_script: Path, launch_args: list[str]) -> None:
    cmd = [str(launch_script)] + launch_args + ["--dry-run"]
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background autopilot for full eval suite.")
    parser.add_argument("--expver", required=True)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--eval-date", default="20230801/20230802")
    parser.add_argument("--eval-number", default="1/2")
    parser.add_argument("--eval-step", default="24/120")
    parser.add_argument("--quaver-first-date", default="20230826")
    parser.add_argument("--quaver-last-date", default="20230827")
    parser.add_argument("--quaver-nmem", default="2")
    parser.add_argument("--spectra-date", default="20230826/to/20230827/by/1")
    parser.add_argument("--spectra-step", default="144")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    launch_script = project_root / "eval/jobs/launch_full_eval_suite.sh"
    run_dir = Path(args.eval_root) / args.expver
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = logs_dir / "autopilot_state.json"

    launch_args = [
        "--expver",
        args.expver,
        "--eval-root",
        args.eval_root,
        "--eval-date",
        args.eval_date,
        "--eval-number",
        args.eval_number,
        "--eval-step",
        args.eval_step,
        "--quaver-first-date",
        args.quaver_first_date,
        "--quaver-last-date",
        args.quaver_last_date,
        "--quaver-nmem",
        args.quaver_nmem,
        "--spectra-date",
        args.spectra_date,
        "--spectra-step",
        args.spectra_step,
    ]
    _regenerate_scripts(launch_script, launch_args)

    gen_dir = run_dir / "jobs"
    jobs: dict[str, JobTrack] = {
        "eval_mars": JobTrack(
            name="eval_mars",
            script=gen_dir / f"eval_mars_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "quaver": JobTrack(
            name="quaver",
            script=gen_dir / f"quaver_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "spectra_compute": JobTrack(
            name="spectra_compute",
            script=gen_dir / f"spectra_compute_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "spectra_plot": JobTrack(
            name="spectra_plot",
            script=gen_dir / f"spectra_plot_{args.expver}.sbatch",
            dependency="spectra_compute",
            max_retries=args.max_retries,
        ),
        "tc_retrieve": JobTrack(
            name="tc_retrieve",
            script=gen_dir / f"tc_retrieve_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "tc_plot": JobTrack(
            name="tc_plot",
            script=gen_dir / f"tc_plot_{args.expver}.sbatch",
            dependency="tc_retrieve",
            max_retries=args.max_retries,
        ),
    }

    for key in ["eval_mars", "quaver", "spectra_compute", "tc_retrieve"]:
        jobs[key].job_id = _submit(jobs[key].script)

    jobs["spectra_plot"].job_id = _submit(
        jobs["spectra_plot"].script, dependency=jobs["spectra_compute"].job_id
    )
    jobs["tc_plot"].job_id = _submit(
        jobs["tc_plot"].script, dependency=jobs["tc_retrieve"].job_id
    )

    _write_state(state_file, jobs, args.expver)
    print(f"Autopilot started for {args.expver}. State file: {state_file}")

    while True:
        for j in jobs.values():
            if not j.job_id:
                continue
            j.state = _job_state(j.job_id)

        if _all_terminal(jobs):
            break

        for j in jobs.values():
            if j.state in TERMINAL_BAD and j.retries < j.max_retries:
                old_job_id = j.job_id
                j.retries += 1
                if j.dependency:
                    dep_id = jobs[j.dependency].job_id
                    j.job_id = _submit(j.script, dependency=dep_id)
                else:
                    j.job_id = _submit(j.script)
                j.state = "PENDING"
                if old_job_id and old_job_id != j.job_id:
                    _cancel(old_job_id)

                if j.name == "spectra_compute":
                    sp = jobs["spectra_plot"]
                    old_sp_job_id = sp.job_id
                    sp.job_id = _submit(sp.script, dependency=j.job_id)
                    sp.state = "PENDING"
                    if old_sp_job_id and old_sp_job_id != sp.job_id:
                        _cancel(old_sp_job_id)
                if j.name == "tc_retrieve":
                    tp = jobs["tc_plot"]
                    old_tp_job_id = tp.job_id
                    tp.job_id = _submit(tp.script, dependency=j.job_id)
                    tp.state = "PENDING"
                    if old_tp_job_id and old_tp_job_id != tp.job_id:
                        _cancel(old_tp_job_id)

        _write_state(state_file, jobs, args.expver)
        time.sleep(max(5, args.poll_seconds))

    _write_state(state_file, jobs, args.expver)
    bad = {k: v.state for k, v in jobs.items() if v.state in TERMINAL_BAD}
    if bad:
        print("Autopilot finished with failures:")
        for k, v in bad.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)
    print("Autopilot finished successfully.")


if __name__ == "__main__":
    main()
