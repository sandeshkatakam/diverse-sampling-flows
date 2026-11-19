#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
usage() { printf 'Usage: bash launchers/train.sh --gpu ID --config FILE [--method NAME] [--resume FILE] [--updates N] [--detach] [--run-id ID] [--dry-run]\n'; }
gpu=""; config="$PROJECT_ROOT/configs/main.yaml"; method=skew_trained; resume=""; updates=""; run_id=""; output_root="$PROJECT_ROOT/outputs"; detach=false; dry_run=false; overrides=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --method) method="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --updates) updates="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --override) overrides+=("$2"); shift 2 ;;
    --detach) detach=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$gpu" ]] || die "training requires --gpu"
require_venv; select_gpu "$gpu"; run_id="${run_id:-$(default_run_id train)}"
command=("$VENV_PYTHON" -m diverse_sampling_flows.cli.train --config "$config" --method "$method" --device cuda:0 --output-root "$output_root" --run-id "$run_id")
[[ -n "$resume" ]] && command+=(--resume "$resume")
[[ -n "$updates" ]] && command+=(--updates "$updates")
command+=("${overrides[@]}")
if [[ "$dry_run" == "true" ]]; then print_environment "$gpu"; print_command "${command[@]}"; exit 0; fi
run_logged "$PROJECT_ROOT/logs/train/$run_id" "$detach" "${command[@]}"
