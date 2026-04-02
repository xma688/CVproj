#!/bin/bash
# 用法:
# bash scripts/eval_3dgs.sh DL3DV-2 PlanA
# bash scripts/eval_3dgs.sh Re10k-1 PlanB
# bash scripts/eval_3dgs.sh 405841_FRONT PlanA

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$ROOT/gaussian-splatting"

SCENE="${1:-DL3DV-2}"
PLAN="${2:-PlanA}"

case "$SCENE" in
  DL3DV-2|Re10k-1|405841_FRONT)
    ;;
  *)
    echo "Unsupported scene: $SCENE"
    echo "Supported scenes: DL3DV-2 | Re10k-1 | 405841_FRONT"
    exit 1
    ;;
esac

case "$PLAN" in
  PlanA|PlanB)
    ;;
  *)
    echo "Unsupported plan: $PLAN"
    echo "Supported plans: PlanA | PlanB"
    exit 1
    ;;
esac

OUT_DIR="$ROOT/outputs/3dgs/$PLAN/$SCENE"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ ! -d "$OUT_DIR" ]; then
  echo "Error: trained model folder not found: $OUT_DIR"
  exit 1
fi

cd "$REPO"

python render.py \
  -m "$OUT_DIR" \
  --skip_train \
  2>&1 | tee "$LOG_DIR/render_test_console.log"

python metrics.py \
  -m "$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/metrics_test_console.log"

{
  echo "========================================"
  echo "SCENE    : $SCENE"
  echo "PLAN     : $PLAN"
  echo "MODEL    : $OUT_DIR"
  echo "NOTE     : render.py ran with --skip_train, so only test split was rendered."
  echo "========================================"
} | tee "$LOG_DIR/eval_meta.txt"