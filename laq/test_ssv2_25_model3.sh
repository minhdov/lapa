mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python test_ssv2_25_model3.py \
  --checkpoint results/results_stage252_rgbfeature_only3/stage252.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_dir /datasets/ssv2_libero_90/stage25_model_3_val_libero10_new/z_depth_val \
  --output_prefix z_depth_val_model3 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model3_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4