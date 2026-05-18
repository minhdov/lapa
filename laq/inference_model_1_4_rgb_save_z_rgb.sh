#!/bin/bash
set -e

GPU_ID=0

CHECKPOINT="results/model_1_4_rgb_ssv2_25k/vae.25000.pt"

BASE_DATA="/datasets/ssv2_libero90"
INPUT_DIR="${BASE_DATA}/shards_rgb"
OUTPUT_BASE="${BASE_DATA}/model_1_4_rgb_ssv2"

mkdir -p "${OUTPUT_BASE}/jsonl"
mkdir -p "${OUTPUT_BASE}/features"
mkdir -p logs

for SHARD in 0 1 2 3
do
  echo "=========================================="
  echo "Running Model 1.4 RGB inference for shard ${SHARD}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_model_1_4_rgb_save_z_rgb.py \
    --input_file "${INPUT_DIR}/frames_train_shard${SHARD}.jsonl" \
    --dist_number 1 \
    --divider 1 \
    --codebook_size 8 \
    --laq_checkpoint "${CHECKPOINT}" \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --unshuffled_jsonl "${OUTPUT_BASE}/jsonl/z_rgb_train_shard${SHARD}_model_1_4.jsonl" \
    --feature_dir "${OUTPUT_BASE}/features/z_rgb_train_shard${SHARD}" \
    --feature_prefix "z_rgb_train_shard${SHARD}_model_1_4_rgb" \
    --feature_part_size 8192 \
    --batch_size 64 \
    --num_workers 4 \
    --save_z_rgb_tokens \
    2>&1 | tee "logs/inference_model_1_4_rgb_shard${SHARD}.log"

  echo "Finished shard ${SHARD}"
done

echo "All 4 shards finished."