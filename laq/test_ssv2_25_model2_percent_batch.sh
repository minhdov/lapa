#!/bin/bash

PERCENTS=(5 10 20 40 60 80)
STEPS_LIST=(0 5000 15000 25000 35000 45000 55000 65000)
GPUS=(0 2 7 8)

run_one_job () {
  local gpu=$1
  local percent=$2
  local steps=$3

  echo "========================================"
  echo "Running percent=${percent}, steps=${steps} on GPU ${gpu}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=${gpu} python test_ssv2_25_model2_percent.py \
    --checkpoint /outputs/results_stage25_final_feature_model2_ssv2_${percent}p0percent/stage25.${steps}.pt \
    --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
    --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
    --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model2_${percent}_percent_${steps}_steps_debug.jsonl \
    --feature_output_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_${percent}_percent_${steps}_steps/z_depth_val \
    --feature_prefix z_depth_val_model2 \
    --batch_size 256 \
    --num_workers 4

  echo "Finished percent=${percent}, steps=${steps} on GPU ${gpu}"
}

job_id=0

for percent in "${PERCENTS[@]}"; do
  for steps in "${STEPS_LIST[@]}"; do

    gpu=${GPUS[$((job_id % 4))]}

    run_one_job ${gpu} ${percent} ${steps} &

    job_id=$((job_id + 1))

    # Sau mỗi 4 jobs thì đợi cả 4 GPU chạy xong rồi mới chạy batch tiếp theo
    if (( job_id % 4 == 0 )); then
      wait
    fi

  done
done

# Đợi các job còn lại nếu tổng số job không chia hết cho 4
wait

echo "All jobs finished."