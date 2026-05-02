mkdir -p logs

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_only.py' \
#   > logs/train_stage25_gpu2_feature_only.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_only.py' \
  > logs/train_stage25_gpu2_feature_only2.log 2>&1 < /dev/null &
