# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs

CUDA_VISIBLE_DEVICES=1 python inference_stage25.py \
  --checkpoint results/results_stage25/stage25.15000.pt \
  --z_rgb_path /datasets/ssv2/nips/z_rgb_train.jsonl \
  --z_depth_path /datasets/ssv2/nips/z_depth_train.jsonl \
  --output_jsonl results_stage25/predictions_debug.jsonl \
  --batch_size 64 \
  --max_batches 10


