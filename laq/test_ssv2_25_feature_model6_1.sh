
CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_feature_model6_1.py \
  --checkpoint results/results_stage25_model2_zrgb_ssv2_libero90_to_zdepth_indices_corl_da/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage25_model_6_1_libero10/z_rgb_val_all_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/stage25_predictions_val_model6_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_6_1_val/z_depth_val \
  --feature_prefix z_depth_val_model6_1 \
  --batch_size 256 \
  --num_workers 4 
