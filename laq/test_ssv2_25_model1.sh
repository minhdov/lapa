mkdir -p logs


CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model1.py \
  --checkpoint results/results_stage25_final/stage25.65000.pt \
  --z_rgb_path /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0.jsonl \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_1_val_libero10_new/z_depth_val \
  --feature_prefix z_depth_val_model1 \
  --batch_size 256 \
  --num_workers 4