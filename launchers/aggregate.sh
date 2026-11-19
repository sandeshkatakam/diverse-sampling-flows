#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage() { printf 'Usage: bash launchers/aggregate.sh --input RAW.jsonl [--input RAW2.jsonl ...] --output TABLE.json [--dry-run]\n'; }
inputs=(); output=""; dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --input) inputs+=("$2"); shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
(( ${#inputs[@]} > 0 )) || die "aggregation requires --input"
[[ -n "$output" ]] || die "aggregation requires --output"
command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.aggregate)
for input in "${inputs[@]}"; do command+=(--input "$input"); done
command+=(--output "$output")
if [[ "$dry_run" == "true" ]]; then print_command "${command[@]}"; exit 0; fi
require_venv
run_id="$(default_run_id aggregate)"
run_logged "$PROJECT_ROOT/logs/aggregate/$run_id" false "${command[@]}"
