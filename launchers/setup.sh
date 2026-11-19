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
  "$ROOT/.venv/bin/python -m pip install --constraint $ROOT/requirements/constraints.txt -e $ROOT[dev]"
  "$ROOT/.venv/bin/python -m pip check"
)
if [[ "$dry_run" == "true" ]]; then
  printf '%s\n' "${commands[@]}"
  exit 0
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3.11 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --constraint "$ROOT/requirements/constraints.txt" -e "$ROOT[dev]"
"$ROOT/.venv/bin/python" -m pip check
"$ROOT/.venv/bin/python" -c 'import torch,diffusers,transformers,omegaconf; print(torch.__version__, torch.version.cuda)'
