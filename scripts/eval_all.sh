#!/usr/bin/env bash
# Evaluate a (mock or real) ProCA checkpoint on the 5 cultures.
# Usage:
#   bash scripts/eval_all.sh --mock
#   bash scripts/eval_all.sh --ckpt artifacts/checkpoints/final.pt
set -euo pipefail
cd "$(dirname "$0")/.."

CULTURES=("China" "Germany" "UK" "Mexico" "Japan")

for c in "${CULTURES[@]}"; do
  echo "================ $c (Standard) ================"
  python -m proca.eval.wvs_eval --culture "$c" --baseline standard "$@"
  echo "================ $c (Cultural) ================"
  python -m proca.eval.wvs_eval --culture "$c" --baseline cultural "$@"
  echo "================ $c (X-Lingual)   ================"
  python -m proca.eval.xling_eval --culture "$c" "$@"
done
