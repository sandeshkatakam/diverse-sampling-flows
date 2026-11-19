#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { printf 'Usage: scripts/record_command.sh OUTPUT_DIR COMMAND...\n' >&2; exit 2; }
output="$1"; shift
mkdir -p "$output"
printf '%q ' "$@" >"$output/command.txt"; printf '\n' >>"$output/command.txt"
set +e
"$@" >"$output/stdout.log" 2>"$output/stderr.log"
status=$?
set -e
printf '%s\n' "$status" >"$output/exit_status"
exit "$status"
