#!/bin/bash
set -e

mkdir -p logs
mkdir -p outputs-corl/stage1_depth_gt
mkdir -p /datasets/ssv2_libero_90/stage1_depth_gt

# shard0 on GPU 0
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2_libero_90/depth_train_shard0.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/corl_stage1_depth/vae.25000_ssv2_libero90_gt.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs-corl/stage1_depth_gt/debug_depth_libero_shard0 \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2_libero_90/stage1_depth_gt/z_depth_train_shard0.jsonl \
    --feature_dir /datasets/ssv2_libero_90/stage1_depth_gt/features/z_depth_train_shard0 \
    --feature_prefix z_depth_train_shard0_stage1 \
    --feature_part_size 8192' \
    > logs/stage1_depth_gt_shard0.log 2>&1 < /dev/null &

# shard1 on GPU 1
nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2_libero_90/depth_train_shard1.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/corl_stage1_depth/vae.25000_ssv2_libero90_gt.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs-corl/stage1_depth_gt/debug_depth_libero_shard1 \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2_libero_90/stage1_depth_gt/z_depth_train_shard1.jsonl \
    --feature_dir /datasets/ssv2_libero_90/stage1_depth_gt/features/z_depth_train_shard1 \
    --feature_prefix z_depth_train_shard1_stage1 \
    --feature_part_size 8192' \
    > logs/stage1_depth_gt_shard1.log 2>&1 < /dev/null &

# shard2 on GPU 2
nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2_libero_90/depth_train_shard2.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/corl_stage1_depth/vae.25000_ssv2_libero90_gt.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs-corl/stage1_depth_gt/debug_depth_libero_shard2 \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2_libero_90/stage1_depth_gt/z_depth_train_shard2.jsonl \
    --feature_dir /datasets/ssv2_libero_90/stage1_depth_gt/features/z_depth_train_shard2 \
    --feature_prefix z_depth_train_shard2_stage1 \
    --feature_part_size 8192' \
    > logs/stage1_depth_gt_shard2.log 2>&1 < /dev/null &

# shard3 on GPU 3
nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2_libero_90/depth_train_shard3.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/corl_stage1_depth/vae.25000_ssv2_libero90_gt.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs-corl/stage1_depth_gt/debug_depth_libero_shard3 \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2_libero_90/stage1_depth_gt/z_depth_train_shard3.jsonl \
    --feature_dir /datasets/ssv2_libero_90/stage1_depth_gt/features/z_depth_train_shard3 \
    --feature_prefix z_depth_train_shard3_stage1 \
    --feature_part_size 8192' \
    > logs/stage1_depth_gt_shard3.log 2>&1 < /dev/null &

echo "Started all 4 stage1_depth_gt shard inference jobs."
echo "Logs:"
echo "  logs/stage1_depth_gt_shard0.log"
echo "  logs/stage1_depth_gt_shard1.log"
echo "  logs/stage1_depth_gt_shard2.log"
echo "  logs/stage1_depth_gt_shard3.log"