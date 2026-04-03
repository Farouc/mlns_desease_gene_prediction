#!/usr/bin/env bash
set -euo pipefail

if type base >/dev/null 2>&1; then
  base
fi

CONFIG=${1:-configs/han.yaml}

for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  run_name="alpha_${alpha}"
  python main.py \
    --config "${CONFIG}" \
    --run-name "${run_name}" \
    --override models.hybrid.alpha="${alpha}" models.hybrid.search_alpha=false
done
