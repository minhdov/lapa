#!/usr/bin/env bash
set -euo pipefail

LIBERO_DEPTH="/datasets/libero_images_ssv2"

OUTPUT_ROOT="/datasets/ssv2_libero_half"
OUTPUT_DEPTH="${OUTPUT_ROOT}/frames_train"

SEED=42

mkdir -p "$OUTPUT_DEPTH"
mkdir -p "$OUTPUT_ROOT"

echo "LIBERO depth : $LIBERO_DEPTH"
echo "Output depth : $OUTPUT_DEPTH"
echo "Seed         : $SEED"
echo

if [ ! -d "$LIBERO_DEPTH" ]; then
  echo "[ERROR] LIBERO_DEPTH does not exist: $LIBERO_DEPTH"
  exit 1
fi

echo "[1/1] Linking 50% LIBERO depth folders per task..."

python - <<PY
from pathlib import Path
from collections import defaultdict
import random
import os
import re

src = Path("$LIBERO_DEPTH")
dst_root = Path("$OUTPUT_DEPTH")
list_file = Path("$OUTPUT_ROOT/selected_libero.txt")
summary_file = Path("$OUTPUT_ROOT/selected_libero_summary.txt")
seed = int("$SEED")

def get_task_name(folder_name: str) -> str:
    """
    Examples:
      KITCHEN_SCENE3_x_demo_0  -> KITCHEN_SCENE3_x
      KITCHEN_SCENE3_x_demo_12 -> KITCHEN_SCENE3_x
    """
    m = re.match(r"^(.*)_demo_\\d+$", folder_name)
    if m:
        return m.group(1)
    return folder_name

dirs = sorted([p for p in src.iterdir() if p.is_dir()])

groups = defaultdict(list)
for d in dirs:
    task = get_task_name(d.name)
    groups[task].append(d)

selected_all = []

print(f"LIBERO total folders : {len(dirs)}")
print(f"LIBERO total tasks   : {len(groups)}")

with summary_file.open("w") as sf:
    for task, items in sorted(groups.items()):
        items = sorted(items)

        rng = random.Random(f"{seed}_{task}")
        rng.shuffle(items)

        n = max(1, len(items) // 2) if items else 0
        selected = items[:n]
        selected_all.extend(selected)

        msg = f"{task}: total={len(items)}, selected={len(selected)}"
        print("[TASK]", msg)
        sf.write(msg + "\\n")

with list_file.open("w") as f:
    for d in selected_all:
        f.write(str(d.resolve()) + "\\n")

for d in selected_all:
    dst = dst_root / f"libero_{d.name}"

    if dst.exists() or dst.is_symlink():
        print(f"[SKIP] {dst} already exists")
        continue

    os.symlink(d.resolve(), dst)

print(f"LIBERO selected folders: {len(selected_all)}")
print(f"Saved selected LIBERO list to: {list_file}")
print(f"Saved LIBERO summary to: {summary_file}")
PY

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

echo
echo "Selected lists:"
echo "$OUTPUT_ROOT/selected_libero.txt"
echo "$OUTPUT_ROOT/selected_libero_summary.txt"