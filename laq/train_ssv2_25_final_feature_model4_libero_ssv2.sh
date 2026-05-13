mkdir -p logs

# CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4.py


mkdir -p logs

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4.py' \
  > logs/train_model4_gpu3_libero_ssv2.log 2>&1 < /dev/null &