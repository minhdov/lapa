mkdir -p logs

# CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature.py


nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature.py' \
  > logs/train_stage25_gpu2_feature_model2_ssv2_libero.log 2>&1 < /dev/null &
