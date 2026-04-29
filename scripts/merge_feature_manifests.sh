python merge_feature_manifests.py \
  --root /datasets/ssv2/nips/features \
  --pattern "z_rgb_train_shard*_0/*_manifest.json" \
  --output /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
  --prefix z_rgb_train_all \
  --check-files