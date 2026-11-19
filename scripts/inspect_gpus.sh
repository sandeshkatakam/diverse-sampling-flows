#!/usr/bin/env bash
set -euo pipefail
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
