
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python test_ssv2_25_model7_1_feature.py \
  --checkpoint results/results_stage25_model4_zrgb_ssv2_libero90_to_zdepth_feature_corl_da/model4.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json  \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_7_1_val_libero10_new/z_depth_val \
  --feature_prefix z_depth_val_model7_1 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model7_1_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 --save_gt --save_rgb



