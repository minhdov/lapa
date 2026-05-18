#!/bin/bash
set -e

OUT_DIR="/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90"
OUT_MANIFEST="${OUT_DIR}/z_rgb_train_mixed_manifest.json"

mkdir -p "${OUT_DIR}"

python merge_two_all_rgb_manifests.py \
  --ssv2_manifest /datasets/ssv2_libero_90/stage2_rgb_ssv2/features/z_rgb_train_all_manifest.json \
  --libero90_manifest /datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_all_manifest.json \
  --out_manifest "${OUT_MANIFEST}" \
  --prefix z_rgb_train_mixed

echo "Done merging SSV2 + LIBERO90 RGB manifests:"
echo "${OUT_MANIFEST}"