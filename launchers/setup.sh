#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  printf 'Usage: bash launchers/setup.sh [--dry-run]\n'
}

dry_run=false
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --dry-run) dry_run=true ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac

if command -v uv >/dev/null 2>&1; then
  commands=(
    "uv venv --python python3.11 $ROOT/.venv"
    "uv pip install --python $ROOT/.venv/bin/python --constraint $ROOT/requirements/constraints.txt -e $ROOT[dev]"
    "uv pip check --python $ROOT/.venv/bin/python"
  )
else
  commands=(
    "python3.11 -m venv $ROOT/.venv"
    "$ROOT/.venv/bin/python -m pip install --constraint $ROOT/requirements/constraints.txt -e $ROOT[dev]"
    "$ROOT/.venv/bin/python -m pip check"
  )
fi
if [[ "$dry_run" == "true" ]]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$ROOT/.uv-cache"
  if [[ ! -x "$ROOT/.venv/bin/python" || ! -f "$ROOT/.venv/bin/activate" ]]; then
    uv venv --clear --python python3.11 "$ROOT/.venv"
  fi
  uv pip install --python "$ROOT/.venv/bin/python" --constraint "$ROOT/requirements/constraints.txt" -e "$ROOT[dev]"
  uv pip check --python "$ROOT/.venv/bin/python"
else
  if [[ ! -x "$ROOT/.venv/bin/python" || ! -f "$ROOT/.venv/bin/activate" ]]; then
    python3.11 -m venv --clear "$ROOT/.venv"
  fi
  "$ROOT/.venv/bin/python" -m pip install --constraint "$ROOT/requirements/constraints.txt" -e "$ROOT[dev]"
  "$ROOT/.venv/bin/python" -m pip check
fi
"$ROOT/.venv/bin/python" -c 'import torch,diffusers,transformers,omegaconf; print(torch.__version__, torch.version.cuda)'
