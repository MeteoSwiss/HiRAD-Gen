# Legacy Operational Playbooks

These playbooks describe pre-refactor operational workflows for the `downscaling-tools` evaluation suite. They are preserved here for the tribal knowledge they contain: TC extreme-tail thresholds (Idalia 980-990 hPa MSLP, 25 m/s wind), `*.stats.json` conventions, autopilot log/PID/state file paths, and host/QoS guidance for `dg`/`ng` queues.

They are **not** authoritative for new work. The post-refactor canonical entry points are:

- Unified CLI: `python -m eval.cli {run,predict,evaluate,scoreboard}`
- Modular prediction generation: `python -m eval.predict.main`
- HPC orchestration templates: `../templates/README.md`
- One-command full suite: `../../FULL_SUITE_PLAYBOOK.md`
- Top-level orientation: `../../../README.md` and `../../../ARCHITECTURE.md`

Do not edit these files in place. If a workflow they describe is still valid, lift it into the canonical docs above and let the legacy copy drift.
