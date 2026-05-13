# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs


# CUDA_VISIBLE_DEVICES=0 python inference_stage25_model4_2.py \
#   --checkpoint results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard0.jsonl \
#   --z_depth_feature_manifest /datasets/libero2/features_depth_stage1/z_depth_train_shard0/z_depth_train_shard0_stage1_manifest.json \
#   --feature_output_dir /datasets/libero2/features_stage25_model4/z_depth_train_shard0 \
#   --feature_prefix z_depth_train_shard0_model4 \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard0_model4_debug.jsonl \
#   --batch_size 64 \
#   --num_workers 4 \

# Shard 0 - GPU 2
CUDA_VISIBLE_DEVICES=0 python inference_stage25_model4_2.py \
  --checkpoint results_model4_depth_rgb_to_zdepth_feature_libero_ssv2/model4.65000.pt \
  --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/libero2/z_depth_train_shard0.jsonl \
  --z_depth_feature_manifest /datasets/libero2/features_depth_stage1/z_depth_train_shard0/z_depth_train_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/libero2/features_stage25_model4_both_libero_ssv2/z_depth_train_shard0 \
  --feature_prefix z_depth_train_shard0_model4 \
  --output_jsonl /datasets/libero2/stage25_predictions_train_shard0_model4_both_libero_ssv2_debug.jsonl \
  --batch_size 256 \
  --num_workers 4


# # Shard 1 - GPU 2
# CUDA_VISIBLE_DEVICES=2 python inference_stage25_model4_2.py \
#   --checkpoint results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard1_0/z_rgb_train_shard1_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard1.jsonl \
#   --z_depth_feature_manifest /datasets/libero2/features_depth_stage1/z_depth_train_shard1/z_depth_train_shard1_stage1_manifest.json \
#   --feature_output_dir /datasets/libero2/features_stage25_model4/z_depth_train_shard1 \
#   --feature_prefix z_depth_train_shard1_model4 \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard1_model4_debug.jsonl \
#   --batch_size 256 \
#   --num_workers 4

# # Shard 2 - GPU 2
# CUDA_VISIBLE_DEVICES=2 python inference_stage25_model4_2.py \
#   --checkpoint results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard2.jsonl \
#   --z_depth_feature_manifest /datasets/libero2/features_depth_stage1/z_depth_train_shard2/z_depth_train_shard2_stage1_manifest.json \
#   --feature_output_dir /datasets/libero2/features_stage25_model4/z_depth_train_shard2 \
#   --feature_prefix z_depth_train_shard2_model4 \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard2_model4_debug.jsonl \
#   --batch_size 256 \
#   --num_workers 4

# # Shard 2 - GPU 2
# CUDA_VISIBLE_DEVICES=2 python inference_stage25_model4_2.py \
#   --checkpoint results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
#   --z_rgb_feature_manifest /datasets/libero_ssv2/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json \
#   --z_depth_path /datasets/libero2/z_depth_train_shard2.jsonl \
#   --z_depth_feature_manifest /datasets/libero2/features_depth_stage1/z_depth_train_shard2/z_depth_train_shard2_stage1_manifest.json \
#   --feature_output_dir /datasets/libero2/features_stage25_model4/z_depth_train_shard2 \
#   --feature_prefix z_depth_train_shard2_model4 \
#   --output_jsonl /datasets/libero2/stage25_predictions_train_shard2_model4_debug.jsonl \
#   --batch_size 256 \
#   --num_workers 4