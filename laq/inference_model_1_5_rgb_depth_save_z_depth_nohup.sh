#!/bin/bash
set -e

mkdir -p logs

nohup bash inference_model_1_5_rgb_depth_save_z_depth.sh \
  > logs/inference_model_1_5_rgb_depth_save_z_depth.log 2>&1 &

echo "Started background job."
echo "PID: $!"
echo "Log: logs/inference_model_1_5_rgb_depth_save_z_depth.log"
echo "Monitor with:"
echo "tail -f logs/inference_model_1_5_rgb_depth_save_z_depth.log"