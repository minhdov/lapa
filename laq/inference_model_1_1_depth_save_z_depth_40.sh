#!/bin/bash
set -e

GPU_ID=0
percent=40
CHECKPOINT="/outputs/results/model_1_1_depth_ssv2_25k_percent_${percent}/vae.25000.pt"

BASE_DATA="/datasets/ssv2_libero90"
INPUT_DIR="${BASE_DATA}/shards_depth"
OUTPUT_BASE="${BASE_DATA}/model_1_1_depth_ssv2_percent_${percent}"

mkdir -p "${OUTPUT_BASE}/jsonl"
mkdir -p "${OUTPUT_BASE}/features"
mkdir -p logs

for SHARD in 0 1 2 3
do
  echo "=========================================="
  echo "Running Model 1.1 depth inference for shard ${SHARD}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_model_1_1_depth_save_z_depth.py \
    --input_file "${INPUT_DIR}/depth_train_shard${SHARD}.jsonl" \
    --dist_number 1 \
    --divider 1 \
    --codebook_size 8 \
    --laq_checkpoint "${CHECKPOINT}" \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --unshuffled_jsonl "${OUTPUT_BASE}/jsonl/z_depth_train_shard${SHARD}_model_1_1.jsonl" \
    --feature_dir "${OUTPUT_BASE}/features/z_depth_train_shard${SHARD}" \
    --feature_prefix "z_depth_train_shard${SHARD}_model_1_1_depth" \
    --feature_part_size 8192 \
    --batch_size 64 \
    --num_workers 4 \
    --debug_save_dir "outputs/debug_model_1_1_depth_shard${SHARD}" \
    --debug_num_samples 10 \
    2>&1 | tee "logs/inference_model_1_1_depth_shard${SHARD}.log"

  echo "Finished shard ${SHARD}"
done

echo "All 4 shards finished."