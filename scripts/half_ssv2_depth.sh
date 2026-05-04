#!/usr/bin/env bash
set -euo pipefail

SSV2_DEPTH="/datasets/ssv2/nips/frames_train"

OUTPUT_ROOT="/datasets/ssv2_libero_half"
OUTPUT_DEPTH="${OUTPUT_ROOT}/frames_train"

SEED=42

mkdir -p "$OUTPUT_DEPTH"
mkdir -p "$OUTPUT_ROOT"

echo "SSV2 depth   : $SSV2_DEPTH"
echo "Output depth : $OUTPUT_DEPTH"
echo "Seed         : $SEED"
echo

if [ ! -d "$SSV2_DEPTH" ]; then
  echo "[ERROR] SSV2_DEPTH does not exist: $SSV2_DEPTH"
  exit 1
fi

echo "[1/1] Linking 50% SSV2 depth folders..."

python - <<PY
from pathlib import Path
import random
import os

src = Path("$SSV2_DEPTH")
dst_root = Path("$OUTPUT_DEPTH")
list_file = Path("$OUTPUT_ROOT/selected_ssv2.txt")
seed = int("$SEED")

dirs = sorted([p for p in src.iterdir() if p.is_dir()])

rng = random.Random(seed)
rng.shuffle(dirs)

n = max(1, len(dirs) // 2) if dirs else 0
selected = dirs[:n]

print(f"SSV2 total folders    : {len(dirs)}")
print(f"SSV2 selected folders : {len(selected)}")

with list_file.open("w") as f:
    for d in selected:
        f.write(str(d.resolve()) + "\n")

for d in selected:
    dst = dst_root / d.name

    if dst.exists() or dst.is_symlink():
        print(f"[SKIP] {dst} already exists")
        continue

    os.symlink(d.resolve(), dst)

print(f"Saved selected SSV2 list to: {list_file}")
PY

echo
echo "Done."
echo "Output folder:"
echo "$OUTPUT_DEPTH"

echo
echo "Number of linked folders:"
find "$OUTPUT_DEPTH" -mindepth 1 -maxdepth 1 -type l | wc -l

echo
echo "Number of image files through symlinks:"
find -L "$OUTPUT_DEPTH" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l

echo
echo "Selected list:"
echo "$OUTPUT_ROOT/selected_ssv2.txt"