#!/bin/bash

mkdir -p /outputs

# 5% trên GPU 0
nohup bash -lc '
CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature_model2_percent.py --data_percent 5 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_5/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_5.log 2>&1 < /dev/null &


# 10% trên GPU 1
nohup bash -lc '
CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model2_percent.py --data_percent 10 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_10/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_10.log 2>&1 < /dev/null &


# 20% trên GPU 2
nohup bash -lc '
CUDA_VISIBLE_DEVICES=6 python train_stage25_sthv2_feature_model2_percent.py --data_percent 20 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_20/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_20.log 2>&1 < /dev/null &


# 40% trên GPU 3
nohup bash -lc '
CUDA_VISIBLE_DEVICES=7 python train_stage25_sthv2_feature_model2_percent.py --data_percent 40 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_40/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_40.log 2>&1 < /dev/null &


# 60% trên GPU 4
nohup bash -lc '
CUDA_VISIBLE_DEVICES=8 python train_stage25_sthv2_feature_model2_percent.py --data_percent 60 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_60/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_60.log 2>&1 < /dev/null &


# 80% trên GPU 5
nohup bash -lc '
CUDA_VISIBLE_DEVICES=9 python train_stage25_sthv2_feature_model2_percent.py --data_percent 80 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_80/jsonl/z_depth_train.jsonl
' > /outputs/train_stage25_sthv2_feature_model2_percent_80.log 2>&1 < /dev/null &


echo "Started 6 independent training jobs:"
echo "  GPU 0: 5%"
echo "  GPU 1: 10%"
echo "  GPU 2: 20%"
echo "  GPU 3: 40%"
echo "  GPU 4: 60%"
echo "  GPU 5: 80%"