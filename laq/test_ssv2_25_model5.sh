

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python test_ssv2_25_model5.py \
  --checkpoint results/results_model5_rgb_to_zdepth_feature/model5.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0_stage1_manifest.json \
  --output_dir /datasets/ssv2_libero_90/stage25_model_5_val_libero10_new/z_depth_val \
  --output_prefix z_depth_val_model5 \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_rgb --save_gt

