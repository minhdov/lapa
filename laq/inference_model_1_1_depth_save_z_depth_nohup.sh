#!/bin/bash
# set -e

# mkdir -p logs

# nohup bash inference_model_1_1_depth_save_z_depth_5.sh \
#   > logs/inference_model_1_1_depth_save_z_depth_5.log 2>&1 &

# # nohup bash inference_model_1_1_depth_save_z_depth_10.sh \
# #   > logs/inference_model_1_1_depth_save_z_depth_10.log 2>&1 &

# nohup bash inference_model_1_1_depth_save_z_depth_20.sh \
#   > logs/inference_model_1_1_depth_save_z_depth_20.log 2>&1 &

# # nohup bash inference_model_1_1_depth_save_z_depth_40.sh \
# #   > logs/inference_model_1_1_depth_save_z_depth_40.log 2>&1 &

# nohup bash inference_model_1_1_depth_save_z_depth_60.sh \
#   > logs/inference_model_1_1_depth_save_z_depth_60.log 2>&1 &

nohup bash inference_model_1_1_depth_save_z_depth_80.sh \
  > logs/inference_model_1_1_depth_save_z_depth_80.log 2>&1 &





echo "Started background job."
echo "PID: $!"
echo "Log: logs/inference_model_1_1_depth_save_z_depth_p10.log"
echo "Monitor with:"
echo "tail -f logs/inference_model_1_1_depth_save_z_depth_p10.log"