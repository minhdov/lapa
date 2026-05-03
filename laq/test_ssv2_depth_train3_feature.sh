
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 python3 inference_sthv2_feature.py \
    --input_file /datasets/ssv2/nips/depth_train_shard3.jsonl \
    --dist_number 1 \
    --codebook_size 8 \
    --laq_checkpoint /checkpoints/lapa/stage1_depth/vae.25000_depth_ssv2.pt \
    --divider 1 \
    --window_size 30 \
    --code_seq_len 4 \
    --layer 8 \
    --repeat_depth_to_3ch 1 \
    --debug_save_dir outputs/debug_depth \
    --debug_num_samples 10 \
    --unshuffled_jsonl /datasets/ssv2/nips/z_depth_train_shard3.jsonl \
    --feature_dir /datasets/ssv2/nips/features_depth_stage1/z_depth_train_shard3 \
    --feature_prefix z_depth_train_shard3_stage1 \
    --feature_part_size 8192

