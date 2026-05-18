#!/bin/bash
set -e

mkdir -p logs

SRC_ROOT="/datasets/ssv2_libero_90/stage2_rgb_libero"
OUT_ROOT="/datasets/ssv2_libero_90/stage2_rgb_libero10"

KEEP_REGEX="^libero_10_"

python filter_z_rgb_feature_by_jsonl.py \
  --source_jsonl "${SRC_ROOT}/z_rgb_train_shard0_0.jsonl" \
  --source_manifest "${SRC_ROOT}/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json" \
  --out_jsonl "${OUT_ROOT}/z_rgb_train_shard0_0.jsonl" \
  --out_feature_dir "${OUT_ROOT}/features/z_rgb_train_shard0_0" \
  --prefix z_rgb_train_shard0_0 \
  --keep_regex "${KEEP_REGEX}" \
  --part_size 8192 \
  > logs/filter_z_rgb_libero10_shard0.log 2>&1

python filter_z_rgb_feature_by_jsonl.py \
  --source_jsonl "${SRC_ROOT}/z_rgb_train_shard1_0.jsonl" \
  --source_manifest "${SRC_ROOT}/features/z_rgb_train_shard1_0/z_rgb_train_shard1_0_manifest.json" \
  --out_jsonl "${OUT_ROOT}/z_rgb_train_shard1_0.jsonl" \
  --out_feature_dir "${OUT_ROOT}/features/z_rgb_train_shard1_0" \
  --prefix z_rgb_train_shard1_0 \
  --keep_regex "${KEEP_REGEX}" \
  --part_size 8192 \
  > logs/filter_z_rgb_libero10_shard1.log 2>&1

python filter_z_rgb_feature_by_jsonl.py \
  --source_jsonl "${SRC_ROOT}/z_rgb_train_shard2_0.jsonl" \
  --source_manifest "${SRC_ROOT}/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json" \
  --out_jsonl "${OUT_ROOT}/z_rgb_train_shard2_0.jsonl" \
  --out_feature_dir "${OUT_ROOT}/features/z_rgb_train_shard2_0" \
  --prefix z_rgb_train_shard2_0 \
  --keep_regex "${KEEP_REGEX}" \
  --part_size 8192 \
  > logs/filter_z_rgb_libero10_shard2.log 2>&1

python filter_z_rgb_feature_by_jsonl.py \
  --source_jsonl "${SRC_ROOT}/z_rgb_train_shard3_0.jsonl" \
  --source_manifest "${SRC_ROOT}/features/z_rgb_train_shard3_0/z_rgb_train_shard3_0_manifest.json" \
  --out_jsonl "${OUT_ROOT}/z_rgb_train_shard3_0.jsonl" \
  --out_feature_dir "${OUT_ROOT}/features/z_rgb_train_shard3_0" \
  --prefix z_rgb_train_shard3_0 \
  --keep_regex "${KEEP_REGEX}" \
  --part_size 8192 \
  > logs/filter_z_rgb_libero10_shard3.log 2>&1

echo "Done filtering RGB features to LIBERO10."
echo "Output: ${OUT_ROOT}"