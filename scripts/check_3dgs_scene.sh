#!/bin/bash
# 用法:
# bash scripts/check_3dgs_scene.sh DL3DV-2
# bash scripts/check_3dgs_scene.sh Re10k-1
# bash scripts/check_3dgs_scene.sh 405841_FRONT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE="${1:-DL3DV-2}"
SCENE_DIR="$ROOT/scenes_3dgs/$SCENE"
TXT_DIR="/tmp/${SCENE}_check_txt"

echo "========================================"
echo "Checking scene: $SCENE"
echo "Scene dir      : $SCENE_DIR"
echo "========================================"

if [ ! -d "$SCENE_DIR" ]; then
  echo "Error: scene directory not found:"
  echo "  $SCENE_DIR"
  exit 1
fi

echo
echo "===== basic structure check ====="

if [ -e "$SCENE_DIR/images" ]; then
  echo "[OK] images exists: $SCENE_DIR/images"
else
  echo "[FAIL] images missing: $SCENE_DIR/images"
fi

if [ -d "$SCENE_DIR/sparse/0" ]; then
  echo "[OK] sparse/0 exists: $SCENE_DIR/sparse/0"
else
  echo "[FAIL] sparse/0 missing: $SCENE_DIR/sparse/0"
fi

if [ -e "$SCENE_DIR/input" ]; then
  echo "[INFO] input exists: $SCENE_DIR/input"
fi

if [ -d "$SCENE_DIR/distorted" ]; then
  echo "[INFO] distorted exists: $SCENE_DIR/distorted"
fi

echo
echo "===== tree -L 2 ====="
if command -v tree >/dev/null 2>&1; then
  tree -L 2 "$SCENE_DIR"
else
  find "$SCENE_DIR" -maxdepth 2 | sort
fi

if [ ! -d "$SCENE_DIR/sparse/0" ]; then
  echo
  echo "Cannot inspect camera model because sparse/0 is missing."
  exit 1
fi

rm -rf "$TXT_DIR"
mkdir -p "$TXT_DIR"

colmap model_converter \
  --input_path "$SCENE_DIR/sparse/0" \
  --output_path "$TXT_DIR" \
  --output_type TXT

echo
echo "===== cameras.txt ====="
head -n 20 "$TXT_DIR/cameras.txt"

CAM_MODEL=$(awk '$1 !~ /^#/ && NF>0 {print $2; exit}' "$TXT_DIR/cameras.txt")
IMG_W=$(awk '$1 !~ /^#/ && NF>0 {print $3; exit}' "$TXT_DIR/cameras.txt")
IMG_H=$(awk '$1 !~ /^#/ && NF>0 {print $4; exit}' "$TXT_DIR/cameras.txt")

echo
echo "===== parsed camera info ====="
echo "Detected camera model : ${CAM_MODEL:-UNKNOWN}"
echo "Image size            : ${IMG_W:-?} x ${IMG_H:-?}"

REG_IMAGES=$(awk '
  BEGIN{cnt=0}
  $1 ~ /^#/ {next}
  NF==0 {next}
  {cnt++}
  END{print int(cnt/2)}
' "$TXT_DIR/images.txt")
PTS3D=$(awk 'BEGIN{cnt=0} $1 !~ /^#/ && NF>0 {cnt++} END{print cnt}' "$TXT_DIR/points3D.txt")

echo "Registered images     : $REG_IMAGES"
echo "Sparse points3D       : $PTS3D"

echo
echo "===== verdict ====="
if [ "$CAM_MODEL" = "PINHOLE" ] || [ "$CAM_MODEL" = "SIMPLE_PINHOLE" ]; then
  echo "[PASS] Camera model is compatible with 3DGS."
  echo "You can try training directly:"
  echo "python train.py -s $SCENE_DIR -m /home/user/cjy/outputs/3dgs/$SCENE --eval"
else
  echo "[WARN] Camera model is $CAM_MODEL"
  echo "This usually means you still need convert.py / undistortion."
fi