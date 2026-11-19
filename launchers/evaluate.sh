#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage() { printf 'Usage: bash launchers/evaluate.sh --gpu ID --method NAME [--checkpoint FILE] [--sets N] [--run-id ID] [--dry-run]\n'; }
gpu=""; method=""; checkpoint=""; sets=""; run_id=""; output_root="$PROJECT_ROOT/outputs"; dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --sets) sets="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$gpu" && -n "$method" ]] || die "evaluation requires --gpu and --method"
require_venv; select_gpu "$gpu"; run_id="${run_id:-$(default_run_id evaluate)}"
command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.evaluate --config "$PROJECT_ROOT/configs/main.yaml" --method "$method" --device cuda:0 --output-root "$output_root" --run-id "$run_id")
[[ -n "$checkpoint" ]] && command+=(--checkpoint "$checkpoint")
[[ -n "$sets" ]] && command+=(--sets "$sets")
if [[ "$dry_run" == "true" ]]; then print_environment "$gpu"; print_command "${command[@]}"; exit 0; fi
run_logged "$PROJECT_ROOT/logs/evaluate/$run_id" false "${command[@]}"
