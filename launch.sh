#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU_ID="${GPU_ID:-0}"
SWEEP_GPUS="${SWEEP_GPUS:-0,1,2,3,4}"
PILOT_RUN_ID="${PILOT_RUN_ID:-pilot-s17}"
PILOT_CHECKPOINT="$ROOT/outputs/train/$PILOT_RUN_ID/best.pt"
dry_run=false

case "${1:-}" in
  --dry-run) dry_run=true ;;
  --help|-h)
    printf 'Usage: GPU_ID=0 SWEEP_GPUS=0,1,2,3,4 bash launch.sh [--dry-run]\n'
    exit 0
    ;;
  "") ;;
  *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

cd "$ROOT"

if [[ "$dry_run" == "true" ]]; then
  bash launchers/setup.sh --dry-run
  printf '%s\n' "$ROOT/.venv/bin/python -m diverse_sampling_flows.cli.cache --config $ROOT/configs/main.yaml --all-encoders"
  bash launchers/validate.sh --stage math --device cpu --dry-run
  bash launchers/validate.sh --stage model --gpu "$GPU_ID" --dry-run
  bash launchers/pilot.sh --gpu "$GPU_ID" --run-id "$PILOT_RUN_ID" --dry-run
  bash launchers/evaluate.sh --gpu "$GPU_ID" --method skew_trained --checkpoint "$PILOT_CHECKPOINT" --sets 256 --refinement --dry-run
  bash launchers/sweep.sh --gpus "$SWEEP_GPUS" --max-concurrency 5
  exit 0
fi

bash launchers/setup.sh
source "$ROOT/.venv/bin/activate"
python -m diverse_sampling_flows.cli.cache --config configs/main.yaml --all-encoders
bash launchers/validate.sh --stage math --device cpu
bash launchers/validate.sh --stage model --gpu "$GPU_ID"
bash launchers/pilot.sh --gpu "$GPU_ID" --run-id "$PILOT_RUN_ID"

if [[ ! -f "$PILOT_CHECKPOINT" ]]; then
  printf 'pilot completed without an eligible best checkpoint: %s\n' "$PILOT_CHECKPOINT" >&2
  exit 1
fi

bash launchers/evaluate.sh --gpu "$GPU_ID" --method skew_trained --checkpoint "$PILOT_CHECKPOINT" --sets 256 --refinement
bash launchers/sweep.sh --gpus "$SWEEP_GPUS" --max-concurrency 5
