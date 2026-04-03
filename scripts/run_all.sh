#!/usr/bin/env bash
set -euo pipefail

if type base >/dev/null 2>&1; then
  base
fi

python main.py --config configs/default.yaml --run-name run_default
python main.py --config configs/han.yaml --run-name run_han
python main.py --config configs/node2vec.yaml --run-name run_node2vec
