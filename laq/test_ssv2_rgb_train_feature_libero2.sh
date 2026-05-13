mkdir -p logs

CUDA_VISIBLE_DEVICES=2 python3 inference_sthv2_feature_rgb.py \
    --input_file /datasets/libero_ssv2/shards_rgb/frames_train_shard0.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/stage1_rgb/laq_openx.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --debug_save_dir outputs/debug_rgb_libero_shard0_mini \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/libero2/z_rgb_train_shard0.jsonl \
    --feature_dir /datasets/libero2/features_rgb_stage1/z_rgb_train_shard0 \
    --feature_prefix z_rgb_train_shard0_stage1 \
    --feature_part_size 8192

