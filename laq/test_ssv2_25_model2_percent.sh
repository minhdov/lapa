
# ssv
#!/bin/bash

PERCENTS=(5 10 20 40 60 80)
STEPS=65000

for percent in "${PERCENTS[@]}"; do
  echo "========================================"
  echo "Running percent=${percent}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=0 python test_ssv2_25_model2_percent.py \
    --checkpoint /outputs/results_stage25_final_feature_model2_ssv2_${percent}p0percent/stage25.${STEPS}.pt \
    --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
    --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
    --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model2_percent_${percent}_debug.jsonl \
    --feature_output_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_${percent}_percent_${STEPS}_steps/z_depth_val \
    --feature_output_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_${percent}_percent_${STEPS}_steps/z_depth_val \
    --feature_prefix z_depth_val_model2 \
    --batch_size 256 \
    --num_workers 4

  echo "Finished percent=${percent}"
done
