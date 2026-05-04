#!/usr/bin/env bash
set -euo pipefail

SSV2_DEPTH="/datasets/ssv2/nips/depth_train"
LIBERO_DEPTH="/datasets/libero_depth_ssv2"
OUTPUT_ROOT="/datasets/ssv2_libero"
OUTPUT_DEPTH="${OUTPUT_ROOT}/depth_train"

mkdir -p "$OUTPUT_DEPTH"

echo "SSV2 depth   : $SSV2_DEPTH"
echo "LIBERO depth : $LIBERO_DEPTH"
echo "Output depth : $OUTPUT_DEPTH"
echo

if [ ! -d "$SSV2_DEPTH" ]; then
  echo "[ERROR] SSV2_DEPTH does not exist: $SSV2_DEPTH"
  exit 1
fi

if [ ! -d "$LIBERO_DEPTH" ]; then
  echo "[ERROR] LIBERO_DEPTH does not exist: $LIBERO_DEPTH"
  exit 1
fi

echo "[1/2] Linking SSV2 depth folders..."
for d in "$SSV2_DEPTH"/*; do
  [ -d "$d" ] || continue

  name="$(basename "$d")"
  dst="$OUTPUT_DEPTH/ssv2_${name}"

  if [ -e "$dst" ] || [ -L "$dst" ]; then
    echo "[SKIP] $dst already exists"
    continue
  fi

  ln -s "$(realpath "$d")" "$dst"
done

echo "[2/2] Linking LIBERO depth folders..."
for d in "$LIBERO_DEPTH"/*; do
  [ -d "$d" ] || continue

  name="$(basename "$d")"
  dst="$OUTPUT_DEPTH/libero_${name}"

  if [ -e "$dst" ] || [ -L "$dst" ]; then
    echo "[SKIP] $dst already exists"
    continue
  fi

  ln -s "$(realpath "$d")" "$dst"
done

echo
echo "Done."
echo "Output folder:"
echo "$OUTPUT_DEPTH"

echo
echo "Number of linked episode folders:"
find "$OUTPUT_DEPTH" -mindepth 1 -maxdepth 1 -type l | wc -l

echo
echo "Number of image files through symlinks:"
find -L "$OUTPUT_DEPTH" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l