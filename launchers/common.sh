#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_venv() {
  [[ -x "$VENV_PYTHON" ]] || die "shared environment missing; run bash launchers/setup.sh"
}

gpu_record() {
  local requested="$1" records information uuid model
  if records="$(nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    awk -F', *' -v id="$requested" '$1 == id {print; found=1} END {exit !found}' <<<"$records"
    return
  fi
  mapfile -t gpu_information < <(find /proc/driver/nvidia/gpus -mindepth 2 -maxdepth 2 -name information -print 2>/dev/null | sort)
  information="${gpu_information[$requested]:-}"
  [[ -n "$information" ]] || return 1
  uuid="$(awk -F': *' '$1 ~ /GPU UUID/ {print $2}' "$information")"
  model="$(awk -F': *' '$1 ~ /Model/ {print $2}' "$information")"
  printf '%s, %s, %s, unavailable, unavailable, unavailable\n' "$requested" "$uuid" "$model"
}

validate_gpu_permission() {
  local requested="$1" token requested_uuid permitted=1
  [[ "$requested" =~ ^[0-7]$ ]] || die "GPU ID must be one of 0 through 7"
  requested_uuid="$(gpu_record "$requested" | cut -d, -f2 | xargs)" || die "GPU $requested is unavailable"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    permitted=0
    IFS=',' read -ra visible_tokens <<<"$CUDA_VISIBLE_DEVICES"
    for token in "${visible_tokens[@]}"; do
      token="$(xargs <<<"$token")"
      if [[ "$token" == "$requested" || "$token" == "$requested_uuid" ]]; then
        permitted=1
      fi
    done
    (( permitted == 1 )) || die "GPU $requested is not permitted by existing CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  fi
}

select_gpu() {
  local requested="$1"
  validate_gpu_permission "$requested"
  export CUDA_VISIBLE_DEVICES="$requested"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export HF_HUB_CACHE="$PROJECT_ROOT/cache/huggingface"
  export HF_HUB_OFFLINE=1
  export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

print_environment() {
  local gpu="$1" record
  record="$(gpu_record "$gpu")"
  printf 'CUDA_VISIBLE_DEVICES=%s CUBLAS_WORKSPACE_CONFIG=%s HF_HUB_CACHE=%q HF_HUB_OFFLINE=1 MPLCONFIGDIR=%q\n' \
    "$CUDA_VISIBLE_DEVICES" "$CUBLAS_WORKSPACE_CONFIG" "$HF_HUB_CACHE" "$MPLCONFIGDIR"
  printf 'server_gpu=%s\n' "$record"
}

print_command() {
  printf 'command='
  printf '%q ' "$@"
  printf '\n'
}

run_logged() {
  local log_directory="$1" detach="$2"
  shift 2
  mkdir -p "$log_directory"
  print_command "$@" >"$log_directory/command.txt"
  {
    printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-}"
    printf 'HF_HUB_CACHE=%s\n' "${HF_HUB_CACHE:-}"
  } >"$log_directory/environment.txt"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" =~ ^[0-7]$ ]]; then
    gpu_record "$CUDA_VISIBLE_DEVICES" >"$log_directory/gpu.csv"
  fi
  if [[ "$detach" == "true" ]]; then
    (
      set +e
      "$@" >"$log_directory/stdout.log" 2>"$log_directory/stderr.log"
      status=$?
      printf '%s\n' "$status" >"$log_directory/exit_status"
      printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$log_directory/finished_at_utc"
      exit "$status"
    ) &
    printf '%s\n' "$!" >"$log_directory/pid"
    printf 'started pid=%s logs=%s\n' "$!" "$log_directory"
  else
    set +e
    "$@" >"$log_directory/stdout.log" 2>"$log_directory/stderr.log" &
    pid=$!
    printf '%s\n' "$pid" >"$log_directory/pid"
    wait "$pid"
    status=$?
    set -e
    printf '%s\n' "$status" >"$log_directory/exit_status"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$log_directory/finished_at_utc"
    printf 'exit_status=%s logs=%s\n' "$status" "$log_directory"
    return "$status"
  fi
}

default_run_id() {
  local kind="$1"
  printf '%s-%s-%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$kind" "$$"
}
