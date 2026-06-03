#!/bin/bash

mkdir -p /outputs



# 5% xong rồi chạy 10% trên GPU 8
nohup bash -lc '
CUDA_VISIBLE_DEVICES=0 python train_stage25_sthv2_feature_model2_percent.py --data_percent 5 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_5/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_5.log 2>&1

CUDA_VISIBLE_DEVICES=0 python train_stage25_sthv2_feature_model2_percent.py --data_percent 10 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_10/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_10.log 2>&1
' > /outputs/train_stage25_sthv2_feature_model2_percent_5_then_10.wrapper.log 2>&1 < /dev/null &


# 20% xong rồi chạy 40% trên GPU 1
nohup bash -lc '
CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature_model2_percent.py --data_percent 20 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_20/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_20.log 2>&1

CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature_model2_percent.py --data_percent 40 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_40/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_40.log 2>&1
' > /outputs/train_stage25_sthv2_feature_model2_percent_20_then_40.wrapper.log 2>&1 < /dev/null &


# 60% xong rồi chạy 80% trên GPU 3
nohup bash -lc '
CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model2_percent.py --data_percent 60 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_60/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_60.log 2>&1

CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model2_percent.py --data_percent 80 \
  --z_depth_path=/datasets/ssv2_libero90/model_1_1_depth_ssv2_percent_80/jsonl/z_depth_train.jsonl \
  > /outputs/train_stage25_sthv2_feature_model2_percent_80.log 2>&1
' > /outputs/train_stage25_sthv2_feature_model2_percent_60_then_80.wrapper.log 2>&1 < /dev/null &

echo "Started 3 training chains:"
echo "  GPU 8: 5%  -> 10%"
echo "  GPU 1: 20% -> 40%"
echo "  GPU 3: 60% -> 80%"