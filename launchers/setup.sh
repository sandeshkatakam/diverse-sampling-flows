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

commands=(
  "python3.11 -m venv $ROOT/.venv"
  "uv pip install --python $ROOT/.venv/bin/python --constraint $ROOT/requirements/constraints.txt -e $ROOT[dev]"
  "uv pip check --python $ROOT/.venv/bin/python"
)
if [[ "$dry_run" == "true" ]]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3.11 -m venv "$ROOT/.venv"
fi
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="$ROOT/.uv-cache"
  uv pip install --python "$ROOT/.venv/bin/python" --constraint "$ROOT/requirements/constraints.txt" -e "$ROOT[dev]"
  uv pip check --python "$ROOT/.venv/bin/python"
else
  "$ROOT/.venv/bin/python" -m pip install --constraint "$ROOT/requirements/constraints.txt" -e "$ROOT[dev]"
  "$ROOT/.venv/bin/python" -m pip check
fi
"$ROOT/.venv/bin/python" -c 'import torch,diffusers,transformers,omegaconf; print(torch.__version__, torch.version.cuda)'
