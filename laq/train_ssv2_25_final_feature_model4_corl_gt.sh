mkdir -p logs

# CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4_corl.py



nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model4_corl_gt.py' \
  > logs/train_model4_zrgb_ssv2_libero90_to_zdepth_feature_gpu3_corl_gt.log 2>&1 < /dev/null &