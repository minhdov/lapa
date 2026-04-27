
mkdir -p logs
CUDA_VISIBLE_DEVICES=1 python3 inference_sthv2.py \
    --input_file /datasets/ssv2/nips/depth_train_shard1.jsonl \
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
    --unshuffled_jsonl /datasets/ssv2/nips/z_depth_train_shard1.jsonl


# mkdir -p logs
# nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=2 \
#   python3 inference_sthv2.py \
#     --input_file /home/linhkastner/philo/datasets/ssv2/depth_val.jsonl \
#     --dist_number 1 \
#     --codebook_size 8 \
#     --laq_checkpoint /home/linhkastner/lapa/LAPA/laq/results/vae.25000.pt \
#     --divider 1 \
#     --window_size 30 \
#     --code_seq_len 4 \
#     --layer 8 \
#     --repeat_depth_to_3ch 1 \
#     --debug_save_dir outputs/debug_depth \
#     --debug_num_samples 10 \
#     --unshuffled_jsonl /home/linhkastner/philo/datasets/ssv2/z_depth_val.jsonl' > logs/lapa_inference_depth_val2.log 2>&1 < /dev/null &
