mkdir -p logs

# CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4_model9_2_corl.py



nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4_model9_2_corl.py' \
  > logs/train_stage25_sthv2_feature_model4_model9_2_corl.log 2>&1 < /dev/null &