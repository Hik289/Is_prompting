#!/usr/bin/env bash
# Run §5 ablations (Tables 2-3 of the paper).
#
# Usage (smoke test):
#   bash scripts/run_ablations.sh --mock --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== dialogue_only (Table 3) ==="
python -m proca.ablations.dialogue_only --config configs/proca_default.yaml "$@"

echo "=== intent_only (Table 3) ==="
python -m proca.ablations.intent_only --config configs/proca_default.yaml "$@"

echo "=== reasoning_only / GSM8K (Table 2) ==="
python -m proca.ablations.reasoning_only --config configs/proca_default.yaml --dataset gsm8k_mock "$@"

echo "=== reasoning_only / MathChat (Table 2) ==="
python -m proca.ablations.reasoning_only --config configs/proca_default.yaml --dataset mathchat_mock "$@"

echo "=== teacher_swap: Qwen3 32B (§5 robustness) ==="
python -m proca.ablations.teacher_swap --teacher qwen3_32b --config configs/proca_default.yaml "$@"
