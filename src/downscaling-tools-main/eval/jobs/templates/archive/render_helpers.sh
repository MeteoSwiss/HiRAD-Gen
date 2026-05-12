#!/bin/bash
# Shared bash helpers for submit_*_flow.sh and submit_scoreboard_chain.sh.
# Source this file from the launcher; do not execute directly.

set_var() {
  local file="$1"
  local var="$2"
  local value="$3"
  python3 - "$file" "$var" "$value" <<PY
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
var = sys.argv[2]
value = sys.argv[3]
pattern = re.compile(rf"^{re.escape(var)}=.*$")
lines = path.read_text().splitlines()
updated = False
for i, line in enumerate(lines):
    if pattern.match(line):
        lines[i] = f"{var}={json.dumps(value)}"
        updated = True
        break
if not updated:
    raise SystemExit(f"Variable not found in {path}: {var}")
path.write_text("\n".join(lines) + "\n")
PY
}

set_sbatch_directive() {
  local file="$1"
  local directive="$2"
  local value="$3"
  python3 - "$file" "$directive" "$value" <<PY
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
directive = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines()
pattern = re.compile(rf"^#SBATCH --{re.escape(directive)}(=|\s).*$")
inserted = False
for i, line in enumerate(lines):
    if pattern.match(line):
        lines[i] = f"#SBATCH --{directive}={value}"
        inserted = True
        break
if not inserted:
    last = max((i for i, line in enumerate(lines) if line.startswith("#SBATCH")), default=0)
    lines.insert(last + 1, f"#SBATCH --{directive}={value}")
path.write_text("\n".join(lines) + "\n")
PY
}

drop_sbatch_directive() {
  local file="$1"
  local directive="$2"
  python3 - "$file" "$directive" <<PY
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
directive = sys.argv[2]
pattern = re.compile(rf"^#SBATCH --{re.escape(directive)}(=|\s).*$")
lines = [line for line in path.read_text().splitlines() if not pattern.match(line)]
path.write_text("\n".join(lines) + "\n")
PY
}

extract_job_id() {
  local submit_output="$1"
  local job_id
  job_id="$(printf "%s\n" "${submit_output}" | awk "{print \$NF}")"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job id from sbatch output: ${submit_output}" >&2
    exit 2
  fi
  printf "%s\n" "${job_id}"
}
