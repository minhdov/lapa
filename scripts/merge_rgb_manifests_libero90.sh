#!/bin/bash
set -e

OUT_MANIFEST="/datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_all_manifest.json"

python merge_rgb_manifests_libero90.py \
  --out_manifest "${OUT_MANIFEST}" \
  --prefix z_rgb_train_all \
  /datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json \
  /datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_shard1_0/z_rgb_train_shard1_0_manifest.json \
  /datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json \
  /datasets/ssv2_libero_90/stage2_rgb_libero90/features/z_rgb_train_shard3_0/z_rgb_train_shard3_0_manifest.json

echo "Done merging LIBERO90 RGB manifests:"
echo "${OUT_MANIFEST}"