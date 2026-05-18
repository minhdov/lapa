mkdir -p logs

CUDA_VISIBLE_DEVICES=3 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2_libero_90/depth_train_shard0.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/corl_stage1_depth/vae.25000_ssv2_libero90.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs-corl/stage1_depth/debug_depth_libero_shard0 \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2_libero_90/stage1_depth/z_depth_train_shard0.jsonl \
    --feature_dir /datasets/ssv2_libero_90/stage1_depth/features/z_depth_train_shard0 \
    --feature_prefix z_depth_train_shard0_stage1 \
    --feature_part_size 8192

