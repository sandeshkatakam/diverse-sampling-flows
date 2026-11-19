#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage() { printf 'Usage: bash launchers/pilot.sh --gpu ID [--detach] [--run-id ID] [--dry-run]\n'; }
gpu=""; run_id=""; output_root="$PROJECT_ROOT/outputs"; detach=false; dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --detach) detach=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$gpu" ]] || die "pilot requires --gpu"
select_gpu "$gpu"; run_id="${run_id:-$(default_run_id pilot)}"
command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.pilot --config "$PROJECT_ROOT/configs/main.yaml" --device cuda:0 --output-root "$output_root" --run-id "$run_id")
if [[ "$dry_run" == "true" ]]; then print_environment "$gpu"; print_command "${command[@]}"; exit 0; fi
require_venv
run_logged "$PROJECT_ROOT/logs/pilot/$run_id" "$detach" "${command[@]}"
