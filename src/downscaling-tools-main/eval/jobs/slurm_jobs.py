from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
TERMINAL_ALL = TERMINAL_OK | TERMINAL_BAD


@dataclass
class JobTrack:
    name: str
    script: Path
    dependency: str | None = None
    job_id: str | None = None
    retries: int = 0
    max_retries: int = 1
    state: str = "PENDING"


def run_checked(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def parse_submitted_job_id(output: str) -> str:
    match = re.search(r"(\d+)$", output.strip())
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {output}")
    return match.group(1)


def submit(run_fn, script: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [str(script)]
    return parse_submitted_job_id(run_fn(cmd))


def cancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], check=False)


def parse_sacct_state(output: str, *, job_id: str) -> str:
    for line in output.splitlines():
        state = line.strip()
        if not state:
            continue
        state = state.split("|", 1)[0].strip().upper()
        if state and state != "UNKNOWN":
            return state
    raise RuntimeError(f"No usable sacct state found for job {job_id}: {output!r}")


def sacct_state(run_fn, job_id: str) -> str:
    out = run_fn(["sacct", "-j", job_id, "--format=State", "-n", "-P"])
    return parse_sacct_state(out, job_id=job_id)


def parse_squeue_state(output: str) -> tuple[str | None, str | None]:
    if not output:
        return None, None
    line = output.splitlines()[0].strip()
    if not line:
        return None, None
    if "|" in line:
        state, reason = line.split("|", 1)
        return state.strip().upper() or None, reason.strip()
    return line.strip().upper() or None, None


def squeue_state(run_fn, job_id: str) -> tuple[str | None, str | None]:
    out = run_fn(["squeue", "-h", "-j", job_id, "-o", "%T|%r"])
    return parse_squeue_state(out)


def job_state(run_fn, job_id: str) -> str:
    sq_state, sq_reason = squeue_state(run_fn, job_id)
    if sq_state:
        if sq_reason and "DEPENDENCYNEVERSATISFIED" in sq_reason.upper():
            return "FAILED"
        return sq_state
    return sacct_state(run_fn, job_id)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def jobs_payload(jobs: dict[str, JobTrack]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "job_id": job.job_id,
            "state": job.state,
            "retries": job.retries,
            "max_retries": job.max_retries,
            "script": str(job.script),
            "dependency": job.dependency,
        }
        for key, job in jobs.items()
    }


def updated_epoch() -> int:
    return int(time.time())
