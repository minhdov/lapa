mkdir -p logs

# CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature_model2_model8_corl.py



nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2_feature_model2_model8_corl.py' \
  > logs/train_stage25_sthv2_feature_model2_model8_corl.log 2>&1 < /dev/null &