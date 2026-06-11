from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from eval.jobs import slurm_jobs
from eval.paths import DEFAULT_EVAL_ROOT

TERMINAL_OK = slurm_jobs.TERMINAL_OK
TERMINAL_BAD = slurm_jobs.TERMINAL_BAD
TERMINAL_ALL = slurm_jobs.TERMINAL_ALL
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Default evaluation strategy: proxy-first.
# Phase 1: 10 total proxy date-step-member predictions by default
#          (canonical 10 date/step pairs × member 1, qos=dg, target ~30 min GPU)
#          + TC extreme comparison.
# Phase 2: Pause — another agent fills the proxy scoreboard and reviews.
# Phase 3: Full 250 predictions (25 bundles × 10 members).
PHASE_PROXY = "proxy"
PHASE_FULL = "full"
JobTrack = slurm_jobs.JobTrack


def _run(cmd: list[str]) -> str:
    return slurm_jobs.run_checked(cmd)


def _submit(script: Path, dependency: str | None = None) -> str:
    return slurm_jobs.submit(_run, script, dependency)


def _cancel(job_id: str) -> None:
    slurm_jobs.cancel(job_id)


def _job_state(job_id: str) -> str:
    return slurm_jobs.job_state(_run, job_id)


def _write_state(state_file: Path, run_id: str, phase: str, jobs: dict[str, JobTrack],
                 *, proxy_verdict: str = "", extra: dict | None = None) -> None:
    payload = {
        "run_id": run_id,
        "phase": phase,
        "proxy_verdict": proxy_verdict,
        "updated_epoch": slurm_jobs.updated_epoch(),
        "jobs": slurm_jobs.jobs_payload(jobs),
    }
    if extra:
        payload.update(extra)
    slurm_jobs.write_json(state_file, payload)


def _validate_safe_name(name: str, *, label: str) -> None:
    if not SAFE_NAME_RE.fullmatch(name):
        raise SystemExit(
            f"{label} contains unsafe characters: {name!r}. "
            "Allowed pattern: [A-Za-z0-9._-]+"
        )


def _validate_eval_root(eval_root: Path) -> None:
    if (eval_root / "logs").is_dir() and (eval_root / "jobs").is_dir():
        raise SystemExit(
            f"Refusing eval root that already looks like a run directory: {eval_root}. "
            "Pass the parent eval root (for example /home/ecm5702/scratch/eval)."
        )


def _poll_jobs(jobs: dict[str, JobTrack], state_file: Path, run_id: str, phase: str,
               poll_seconds: int, *, proxy_verdict: str = "") -> dict[str, str]:
    """Poll until all jobs reach a terminal state. Returns {name: state}."""
    while True:
        for j in jobs.values():
            if j.job_id:
                j.state = _job_state(j.job_id)

        if all(j.state in TERMINAL_ALL for j in jobs.values()):
            break

        # Retry logic: for two-job chains (predict→eval), retry predict and relink eval.
        job_list = list(jobs.values())
        if len(job_list) == 2:
            p, e = job_list[0], job_list[1]
            if p.state in TERMINAL_BAD and p.retries < p.max_retries:
                old_p, old_e = p.job_id, e.job_id
                p.retries += 1
                p.job_id = _submit(p.script)
                p.state = "PENDING"
                e.job_id = _submit(e.script, dependency=p.job_id)
                e.state = "PENDING"
                if old_p and old_p != p.job_id:
                    _cancel(old_p)
                if old_e and old_e != e.job_id:
                    _cancel(old_e)

            if e.state in TERMINAL_BAD and e.retries < e.max_retries and p.state in TERMINAL_OK:
                old_e = e.job_id
                e.retries += 1
                e.job_id = _submit(e.script, dependency=p.job_id)
                e.state = "PENDING"
                if old_e and old_e != e.job_id:
                    _cancel(old_e)

        _write_state(state_file, run_id, phase, jobs, proxy_verdict=proxy_verdict)
        time.sleep(max(5, poll_seconds))

    _write_state(state_file, run_id, phase, jobs, proxy_verdict=proxy_verdict)
    return {k: v.state for k, v in jobs.items()}


def _run_proxy_tc_compare(project_root: Path, run_dir: Path, eval_root: Path) -> str:
    """Run proxy TC comparison and return verdict (PASS/WARN/FAIL/SKIPPED)."""
    anchor_json = eval_root / "anchor_tc_extremes.json"
    if not anchor_json.exists():
        print(f"No anchor TC extremes at {anchor_json} — skipping TC comparison.")
        return "SKIPPED"

    pred_dir = run_dir / "predictions"
    out_json = run_dir / "proxy_tc_compare.json"

    cmd = [
        sys.executable, "-m", "eval.jobs.proxy_tc_compare",
        "--proxy-predictions-dir", str(pred_dir),
        "--anchor-json", str(anchor_json),
        "--out-json", str(out_json),
        "--support-mode", "native",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(project_root), capture_output=True, text=True, timeout=600,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.TimeoutExpired:
        raise RuntimeError("TC comparison timed out after 600s")
    except Exception as exc:
        raise RuntimeError(f"TC comparison failed: {exc}") from exc

    if not out_json.exists():
        raise RuntimeError(f"TC comparison did not write verdict JSON: {out_json}")

    try:
        compare_data = json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"TC comparison wrote invalid JSON: {out_json}") from exc

    verdict = str(compare_data.get("overall_verdict", "")).strip().upper()
    if not verdict:
        raise RuntimeError(f"TC comparison JSON missing overall_verdict: {out_json}")
    if result.returncode != 0 and verdict not in {"PASS", "WARN", "FAIL", "SKIPPED"}:
        raise RuntimeError(
            f"TC comparison exited with {result.returncode} and returned unusable verdict {verdict!r}"
        )
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Autopilot for predictions→eval pipeline.\n\n"
            "Default strategy (proxy-first):\n"
            "  Phase 1: Run the canonical 10 proxy date/step pairs with proxy member scope "
            "defaulting to member 1 only (10 total predictions, qos=dg, target ~30 min GPU) "
            "+ TC comparison.\n"
            "  Phase 2: STOP — review the proxy scoreboard (another agent).\n"
            "  Phase 3: Rerun with --continue-full to submit full 250 predictions.\n\n"
            "To skip proxy and go straight to full 250: --full-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--input-root", default="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia")
    ap.add_argument("--ckpt-id", default="4a5b2f1b24b84c52872bfcec1410b00f")
    ap.add_argument("--name-ckpt", default="",
                     help="Explicit checkpoint path. Overrides --ckpt-id.")
    ap.add_argument(
        "--extra-args-json",
        default="",
        help="Optional sampler override JSON passed through to the generated prediction jobs.",
    )
    ap.add_argument("--predict-qos", default="dg")
    ap.add_argument("--predict-time", default="00:30:00")
    ap.add_argument("--predict-cpus", default="32")
    ap.add_argument("--predict-mem", default="256G")
    ap.add_argument("--predict-gpus", default="1")
    ap.add_argument(
        "--proxy-members",
        default="1",
        help=(
            "CSV member ids for the proxy phase. Default '1' keeps the proxy gate at "
            "10 total date-step-member predictions across the canonical 10 date/step pairs."
        ),
    )
    ap.add_argument("--eval-qos", default="nf")
    ap.add_argument("--eval-time", default="08:00:00")
    ap.add_argument("--eval-cpus", default="8")
    ap.add_argument("--eval-mem", default="64G")
    ap.add_argument(
        "--proxy-scoreboard-artifacts",
        action="store_true",
        help=(
            "During proxy phase, also write strict proxy TC JSON/PDF and spectra summary "
            "using the canonical proxy bundle helper path."
        ),
    )
    ap.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help="Allow reuse of an existing run directory (explicitly unsafe).",
    )
    ap.add_argument("--resume", action="store_true", default=True)

    # Phase control
    phase_group = ap.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--full-only",
        action="store_true",
        help="Skip proxy phase, go straight to full 250 predictions (old behavior).",
    )
    phase_group.add_argument(
        "--continue-full",
        action="store_true",
        help="Continue from completed proxy to full 250 predictions. "
             "Use after reviewing the proxy scoreboard.",
    )
    args = ap.parse_args()

    _validate_safe_name(args.run_id, label="run-id")
    eval_root = Path(args.eval_root).expanduser().resolve()
    _validate_eval_root(eval_root)
    run_dir = eval_root / args.run_id
    project_root = Path(__file__).resolve().parents[2]

    # ── Determine which phase to run ──────────────────────────────────────────
    if args.full_only:
        phases = [PHASE_FULL]
    elif args.continue_full:
        phases = [PHASE_FULL]
        # Must have existing run dir from proxy phase
        if not run_dir.exists():
            raise SystemExit(
                f"No run directory at {run_dir}. Run proxy phase first (without --continue-full)."
            )
        args.allow_existing_run_dir = True
    else:
        # Default: proxy only. User/agent reviews, then reruns with --continue-full.
        phases = [PHASE_PROXY]

    if run_dir.exists() and not args.allow_existing_run_dir:
        raise SystemExit(
            f"Run directory already exists: {run_dir}. "
            "Refusing silent reuse; use a new run-id or pass --allow-existing-run-dir explicitly."
        )

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = logs_dir / "autopilot_predictions_state.json"

    ckpt_flag_args = []
    if args.name_ckpt:
        ckpt_flag_args = ["--name-ckpt", args.name_ckpt]
    else:
        ckpt_flag_args = ["--ckpt-id", args.ckpt_id]

    # ── Phase: PROXY ──────────────────────────────────────────────────────────
        if PHASE_PROXY in phases:
            proxy_launch = project_root / "eval/jobs/launch_proxy_eval.sh"

        proxy_args = [
            str(proxy_launch),
            "--run-id", args.run_id,
            "--eval-root", args.eval_root,
            "--input-root", args.input_root,
            *ckpt_flag_args,
            "--predict-qos", args.predict_qos,
            "--predict-time", args.predict_time,
            "--predict-cpus", args.predict_cpus,
            "--predict-mem", args.predict_mem,
            "--predict-gpus", str(args.predict_gpus),
            "--members", args.proxy_members,
            "--eval-qos", args.eval_qos,
            "--eval-time", args.eval_time,
            "--eval-cpus", args.eval_cpus,
            "--eval-mem", args.eval_mem,
            "--dry-run",
        ]
        if args.extra_args_json:
            proxy_args += ["--extra-args-json", args.extra_args_json]
        if args.proxy_scoreboard_artifacts:
            proxy_args.append("--write-scoreboard-artifacts")
        if args.allow_existing_run_dir:
            proxy_args.append("--allow-existing-run-dir")
        subprocess.check_call(proxy_args)

        gen_dir = run_dir / "jobs"
        proxy_jobs = {
            "proxy_predict": JobTrack(
                name="proxy_predict",
                script=gen_dir / f"predict_proxy_{args.run_id}.sbatch",
                max_retries=args.max_retries,
            ),
            "proxy_eval": JobTrack(
                name="proxy_eval",
                script=gen_dir / f"eval_proxy_{args.run_id}.sbatch",
                dependency="proxy_predict",
                max_retries=args.max_retries,
            ),
        }

        # Check for resume
        resumed = False
        if args.resume and state_file.exists():
            try:
                prev = json.loads(state_file.read_text())
                if prev.get("phase") == PHASE_PROXY:
                    for k, j in proxy_jobs.items():
                        pj = prev.get("jobs", {}).get(k, {})
                        if pj.get("job_id"):
                            j.job_id = str(pj["job_id"])
                        if "retries" in pj:
                            j.retries = int(pj["retries"])
                    if proxy_jobs["proxy_predict"].job_id and proxy_jobs["proxy_eval"].job_id:
                        resumed = True
            except Exception:
                resumed = False

        if not resumed:
            proxy_jobs["proxy_predict"].job_id = _submit(proxy_jobs["proxy_predict"].script)
            proxy_jobs["proxy_eval"].job_id = _submit(
                proxy_jobs["proxy_eval"].script,
                dependency=proxy_jobs["proxy_predict"].job_id,
            )
            print(f"[PROXY] Submitted proxy predict+eval for {args.run_id}")
        else:
            print(f"[PROXY] Resumed proxy jobs for {args.run_id}")

        _write_state(state_file, args.run_id, PHASE_PROXY, proxy_jobs)

        # Poll proxy jobs
        final_states = _poll_jobs(
            proxy_jobs, state_file, args.run_id, PHASE_PROXY, args.poll_seconds,
        )

        bad = {k: v for k, v in final_states.items() if v in TERMINAL_BAD}
        if bad:
            print("[PROXY] Proxy phase failed:")
            for k, v in bad.items():
                print(f"  {k}: {v}")
            _write_state(state_file, args.run_id, PHASE_PROXY, proxy_jobs, proxy_verdict="PROXY_JOBS_FAILED")
            raise SystemExit(1)

        # Proxy jobs succeeded — run TC comparison
        print("[PROXY] Proxy predictions + eval complete. Running TC comparison...")
        verdict = _run_proxy_tc_compare(project_root, run_dir, eval_root)
        _write_state(state_file, args.run_id, PHASE_PROXY, proxy_jobs, proxy_verdict=verdict)

        # Report and stop
        print()
        print("=" * 70)
        print(f"  PROXY PHASE COMPLETE — TC verdict: {verdict}")
        print(f"  Run directory: {run_dir}")
        print(f"  State file:    {state_file}")
        print()
        print("  Next steps:")
        print("    1. Review the proxy scoreboard (scoreboard agent).")
        print("    2. If satisfied, continue to full 250 predictions:")
        print(f"       python -m eval.jobs.autopilot_predictions \\")
        print(f"         --run-id {args.run_id} --continue-full \\")
        print(f"         {' '.join(ckpt_flag_args)}")
        print("=" * 70)
        return

    # ── Phase: FULL ───────────────────────────────────────────────────────────
    if PHASE_FULL in phases:
        full_launch = project_root / "eval/jobs/launch_predictions_eval_suite.sh"

        full_args = [
            str(full_launch),
            "--run-id", args.run_id,
            "--eval-root", args.eval_root,
            "--input-root", args.input_root,
            *ckpt_flag_args,
            "--predict-qos", args.predict_qos,
            "--predict-time", args.predict_time,
            "--predict-cpus", args.predict_cpus,
            "--predict-mem", args.predict_mem,
            "--predict-gpus", str(args.predict_gpus),
            "--eval-qos", args.eval_qos,
            "--eval-time", args.eval_time,
            "--eval-cpus", args.eval_cpus,
            "--eval-mem", args.eval_mem,
            "--allow-existing-run-dir",
            "--dry-run",
        ]
        subprocess.check_call(full_args)

        gen_dir = run_dir / "jobs"
        full_jobs = {
            "predict25": JobTrack(
                name="predict25",
                script=gen_dir / f"predict25_{args.run_id}.sbatch",
                max_retries=args.max_retries,
            ),
            "eval25": JobTrack(
                name="eval25",
                script=gen_dir / f"eval25_{args.run_id}.sbatch",
                dependency="predict25",
                max_retries=args.max_retries,
            ),
        }

        # Check for resume
        resumed = False
        if args.resume and state_file.exists():
            try:
                prev = json.loads(state_file.read_text())
                if prev.get("phase") == PHASE_FULL:
                    for k, j in full_jobs.items():
                        pj = prev.get("jobs", {}).get(k, {})
                        if pj.get("job_id"):
                            j.job_id = str(pj["job_id"])
                        if "retries" in pj:
                            j.retries = int(pj["retries"])
                    if full_jobs["predict25"].job_id and full_jobs["eval25"].job_id:
                        resumed = True
            except Exception:
                resumed = False

        if not resumed:
            full_jobs["predict25"].job_id = _submit(full_jobs["predict25"].script)
            full_jobs["eval25"].job_id = _submit(
                full_jobs["eval25"].script,
                dependency=full_jobs["predict25"].job_id,
            )
            print(f"[FULL] Submitted full predict25+eval25 for {args.run_id}")
        else:
            print(f"[FULL] Resumed full jobs for {args.run_id}")

        _write_state(state_file, args.run_id, PHASE_FULL, full_jobs)

        # Poll full jobs
        final_states = _poll_jobs(
            full_jobs, state_file, args.run_id, PHASE_FULL, args.poll_seconds,
        )

        _write_state(state_file, args.run_id, PHASE_FULL, full_jobs)
        bad = {k: v for k, v in final_states.items() if v in TERMINAL_BAD}
        if bad:
            print("[FULL] Full evaluation finished with failures:")
            for k, v in bad.items():
                print(f"  {k}: {v}")
            raise SystemExit(1)

        print("[FULL] Full 250-prediction evaluation completed successfully.")


if __name__ == "__main__":
    main()
