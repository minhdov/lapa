

CUDA_VISIBLE_DEVICES=0 python test_ssv2_25_model2.py \
  --checkpoint results/results_stage25_final_feature_model2_ssv2_libero_ssv2/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model2_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_new/z_depth_val \
  --feature_prefix z_depth_val_model2 \
  --batch_size 256 \
  --num_workers 4 


