#!/bin/bash
# 用法:
# bash scripts/train_3dgs.sh DL3DV-2 PlanA
# bash scripts/train_3dgs.sh Re10k-1 PlanB
# bash scripts/train_3dgs.sh 405841_FRONT PlanA

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

SRC_DIR="$ROOT/scenes_3dgs/$SCENE"
OUT_DIR="$ROOT/outputs/3dgs/$PLAN/$SCENE"
LOG_DIR="$OUT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ ! -d "$SRC_DIR" ]; then
  echo "Error: source scene not found: $SRC_DIR"
  exit 1
fi

START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
START_EPOCH="$(date +%s)"

{
  echo "========================================"
  echo "SCENE          : $SCENE"
  echo "PLAN           : $PLAN"
  echo "SRC_DIR        : $SRC_DIR"
  echo "OUT_DIR        : $OUT_DIR"
  echo "START_TIME     : $START_TIME"
  echo "========================================"
} | tee "$LOG_DIR/train_meta.txt"

cd "$REPO"

python train.py \
  -s "$SRC_DIR" \
  -m "$OUT_DIR" \
  --eval \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 1000 3000 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  2>&1 | tee "$LOG_DIR/train_console.log"

END_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
END_EPOCH="$(date +%s)"
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))

{
  echo
  echo "END_TIME       : $END_TIME"
  echo "ELAPSED_SEC    : $ELAPSED_SEC"
} | tee -a "$LOG_DIR/train_meta.txt"