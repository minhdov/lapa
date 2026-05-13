CUDA_VISIBLE_DEVICES=1 python inference_stage25_feature_save_model2.py \
  --checkpoint results_stage25_final_feature_model2_ssv2_libero_ssv2/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/libero2/z_depth_train_shard0.jsonl \
  --output_jsonl /datasets/libero2/stage25_predictions_train_shard0_model2_both_libero_ssv2_debug.jsonl \
  --feature_output_dir /datasets/libero2/features_stage25_model2_both_libero_ssv2/z_depth_train_shard0 \
  --feature_prefix z_depth_train_shard0_model2 \
  --batch_size 256 \
  --num_workers 4 \
#   --max_batches 5

# # Shard 1
# CUDA_VISIBLE_DEVICES=0 python inference_stage25_feature_save_model2.py \
#   --checkpoint results_stage25_final_feature/stage25.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard1_0/z_rgb_train_shard1_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard1.jsonl \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard1_debug.jsonl \
#   --feature_output_dir /datasets/libero2/features_stage25_model2/z_depth_train_shard1 \
#   --feature_prefix z_depth_train_shard1_model2 \
#   --batch_size 256 \
#   --num_workers 4

# # Shard 2
# CUDA_VISIBLE_DEVICES=0 python inference_stage25_feature_save_model2.py \
#   --checkpoint results_stage25_final_feature/stage25.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard2.jsonl \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard2_debug.jsonl \
#   --feature_output_dir /datasets/libero2/features_stage25_model2/z_depth_train_shard2 \
#   --feature_prefix z_depth_train_shard2_model2 \
#   --batch_size 256 \
#   --num_workers 4

#   # Shard 3
# CUDA_VISIBLE_DEVICES=0 python inference_stage25_feature_save_model2.py \
#   --checkpoint results_stage25_final_feature/stage25.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard3_0/z_rgb_train_shard3_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard3.jsonl \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard3_debug.jsonl \
#   --feature_output_dir /datasets/libero2/features_stage25_model2/z_depth_train_shard3 \
#   --feature_prefix z_depth_train_shard3_model2 \
#   --batch_size 256 \
#   --num_workers 4