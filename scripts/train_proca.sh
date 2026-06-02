#!/usr/bin/env bash
# Train ProCA from scratch.
#
# Dry-run smoke test (CPU, mocked weights and data):
#   bash scripts/train_proca.sh --mock --dry-run
#
# Production-style invocation (assumes real WVS CSV + scenarios are present):
#   bash scripts/train_proca.sh --config configs/proca_default.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

python -m proca.train --config configs/proca_default.yaml "$@"
