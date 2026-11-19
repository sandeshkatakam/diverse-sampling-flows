#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

usage() {
  printf 'Usage: bash launchers/validate.sh --stage math --device cpu [--dry-run]\n'
  printf '       bash launchers/validate.sh --stage model --gpu ID [--output-root DIR] [--run-id ID] [--dry-run]\n'
}

stage=""; device=""; gpu=""; output_root="$PROJECT_ROOT/outputs"; run_id=""; dry_run=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --stage) stage="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "$stage" == "math" || "$stage" == "model" ]] || die "--stage must be math or model"
require_venv
run_id="${run_id:-$(default_run_id "validate-$stage")}" 
if [[ "$stage" == "math" ]]; then
  [[ "${device:-cpu}" == "cpu" ]] || die "math validation device must be cpu"
  command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.validate --config "$PROJECT_ROOT/configs/main.yaml" --stage math --device cpu --output-root "$output_root" --run-id "$run_id")
else
  [[ -n "$gpu" ]] || die "model validation requires --gpu"
  select_gpu "$gpu"
  command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.validate --config "$PROJECT_ROOT/configs/main.yaml" --stage model --device cuda:0 --output-root "$output_root" --run-id "$run_id")
fi
if [[ "$dry_run" == "true" ]]; then
  [[ "$stage" == "model" ]] && print_environment "$gpu"
  print_command "${command[@]}"
  exit 0
fi
run_logged "$PROJECT_ROOT/logs/validation/$run_id" false "${command[@]}"
