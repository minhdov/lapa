
mkdir -p logs

# CUDA_VISIBLE_DEVICES=1 python train_model_1_1_depth_percent.py --data_percent 5

#!/bin/bash

# 20% xong rồi chạy 10%
nohup bash -lc '
CUDA_VISIBLE_DEVICES=1 python train_model_1_1_depth_percent.py --data_percent 20 \
  > /outputs/train_model_1_1_depth_percent_20.log 2>&1

CUDA_VISIBLE_DEVICES=1 python train_model_1_1_depth_percent.py --data_percent 10 \
  > /outputs/train_model_1_1_depth_percent_10.log 2>&1
' > /outputs/train_model_1_1_depth_percent_20_then_10.wrapper.log 2>&1 < /dev/null &


# 40% xong rồi chạy 60%
nohup bash -lc '
CUDA_VISIBLE_DEVICES=3 python train_model_1_1_depth_percent.py --data_percent 40 \
  > /outputs/train_model_1_1_depth_percent_40.log 2>&1

CUDA_VISIBLE_DEVICES=3 python train_model_1_1_depth_percent.py --data_percent 60 \
  > /outputs/train_model_1_1_depth_percent_60.log 2>&1
' > /outputs/train_model_1_1_depth_percent_40_then_60.wrapper.log 2>&1 < /dev/null &


# 80% chạy riêng
nohup bash -lc '
CUDA_VISIBLE_DEVICES=8 python train_model_1_1_depth_percent.py --data_percent 80
' > /outputs/train_model_1_1_depth_percent_80.log 2>&1 < /dev/null &