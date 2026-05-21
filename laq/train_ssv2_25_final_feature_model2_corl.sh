# mkdir -p logs

# CUDA_VISIBLE_DEVICES=0 python train_stage25_sthv2_feature_model2_corl.py


mkdir -p logs

nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python train_stage25_sthv2_feature_model2_corl.py' \
  > logs/train_model2_zrgb_ssv2_libero90_to_zdepth_indices_gpu0_corl.log 2>&1 < /dev/null &