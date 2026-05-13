mkdir -p logs

CUDA_VISIBLE_DEVICES=2 python3 inference_sthv2_feature.py \
    --input_file /datasets/libero_ssv2/depth_train_shard1.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs/debug_depth_libero_shard1_mini \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/libero2/z_depth_train_shard1.jsonl \
    --feature_dir /datasets/libero2/features_depth_stage1/z_depth_train_shard1 \
    --feature_prefix z_depth_train_shard1_stage1 \
    --feature_part_size 8192

# mkdir -p logs

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 inference_sthv2_feature.py \
#     --input_file /datasets/libero_ssv2/depth_train_shard0.jsonl \
#     --dist_number 1 \
#     --codebook_size 8 \
#     --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
#     --divider 1 \
#     --window_size 30 \
#     --code_seq_len 4 \
#     --layer 8 \
#     --repeat_depth_to_3ch 1 \
#     --debug_save_dir outputs/debug_depth_libero_shard0 \
#     --debug_num_samples 10 \
#     --unshuffled_jsonl /datasets/libero_ssv2/z_depth_train_shard0.jsonl \
#     --feature_dir /datasets/libero_ssv2/features_depth_stage1/z_depth_train_shard0 \
#     --feature_prefix z_depth_train_shard0_stage1 \
#     --feature_part_size 8192' \
#     > logs/libero_depth_feature_shard0.log 2>&1 < /dev/null &

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python3 inference_sthv2_feature.py \
#     --input_file /datasets/libero_ssv2/depth_train_shard1.jsonl \
#     --dist_number 1 \
#     --codebook_size 8 \
#     --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
#     --divider 1 \
#     --window_size 30 \
#     --code_seq_len 4 \
#     --layer 8 \
#     --repeat_depth_to_3ch 1 \
#     --debug_save_dir outputs/debug_depth_libero_shard1 \
#     --debug_num_samples 10 \
#     --unshuffled_jsonl /datasets/libero_ssv2/z_depth_train_shard1.jsonl \
#     --feature_dir /datasets/libero_ssv2/features_depth_stage1/z_depth_train_shard1 \
#     --feature_prefix z_depth_train_shard1_stage1 \
#     --feature_part_size 8192' \
#     > logs/libero_depth_feature_shard1.log 2>&1 < /dev/null &

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python3 inference_sthv2_feature.py \
#     --input_file /datasets/libero_ssv2/depth_train_shard2.jsonl \
#     --dist_number 1 \
#     --codebook_size 8 \
#     --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
#     --divider 1 \
#     --window_size 30 \
#     --code_seq_len 4 \
#     --layer 8 \
#     --repeat_depth_to_3ch 1 \
#     --debug_save_dir outputs/debug_depth_libero_shard2 \
#     --debug_num_samples 10 \
#     --unshuffled_jsonl /datasets/libero_ssv2/z_depth_train_shard2.jsonl \
#     --feature_dir /datasets/libero_ssv2/features_depth_stage1/z_depth_train_shard2 \
#     --feature_prefix z_depth_train_shard2_stage1 \
#     --feature_part_size 8192' \
#     > logs/libero_depth_feature_shard2.log 2>&1 < /dev/null &

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python3 inference_sthv2_feature.py \
#     --input_file /datasets/libero_ssv2/depth_train_shard3.jsonl \
#     --dist_number 1 \
#     --codebook_size 8 \
#     --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
#     --divider 1 \
#     --window_size 30 \
#     --code_seq_len 4 \
#     --layer 8 \
#     --repeat_depth_to_3ch 1 \
#     --debug_save_dir outputs/debug_depth_libero_shard3 \
#     --debug_num_samples 10 \
#     --unshuffled_jsonl /datasets/libero_ssv2/z_depth_train_shard3.jsonl \
#     --feature_dir /datasets/libero_ssv2/features_depth_stage1/z_depth_train_shard3 \
#     --feature_prefix z_depth_train_shard3_stage1 \
#     --feature_part_size 8192' \
#     > logs/libero_depth_feature_shard3.log 2>&1 < /dev/null &