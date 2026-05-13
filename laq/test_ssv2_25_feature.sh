# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs

# CUDA_VISIBLE_DEVICES=1 python inference_stage25_feature.py \
#   --checkpoint results_stage25_final_feature/stage25.65000.pt \
#   --z_rgb_feature_manifest /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
#   --z_depth_path /datasets/ssv2/nips/z_depth_train.jsonl \
#   --output_jsonl /datasets/ssv2/nips/stage25_predictions_train.jsonl \
#   --batch_size 64 \
#   --num_workers 4 \
#   --max_batches 5


mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python inference_stage25_feature.py \
  --checkpoint results_stage25_final_feature/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/libero2/z_depth_train_shard0.jsonl \
  --output_jsonl /datasets/libero2/stage25_predictions_train_shard0_debug_model2.jsonl \
  --batch_size 64 \
  --num_workers 4 \
  --max_batches 5
