# python merge_feature_manifests.py \
#   --root /datasets/ssv2/nips/features \
#   --pattern "z_rgb_train_shard*_0/*_manifest.json" \
#   --output /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
#   --prefix z_rgb_train_all \
#   --check-files

# cd /workspace/lapa

# python scripts/merge_feature_manifests.py \
#   --root /datasets/ssv2/nips/features_depth_stage1 \
#   --pattern "z_depth_train_shard*/z_depth_train_shard*_stage1_manifest.json" \
#   --output /datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json \
#   --prefix z_depth_train_stage1 \
#   --check-files

python merge_feature_manifests.py \
  --root /datasets/ssv2/nips/model_1_5_rgb_depth_ssv2/features \
  --pattern "z_depth_train_shard*/z_depth_train_shard*_model_1_5_rgb_depth_manifest.json" \
  --output /datasets/ssv2/nips/model_1_5_rgb_depth_ssv2/features/z_depth_train_model_1_5_rgb_depth_manifest.json \
  --prefix z_depth_train_stage1 \
  --check-files

  # z_depth_train_shard0_model_1_5_rgb_depth_manifest