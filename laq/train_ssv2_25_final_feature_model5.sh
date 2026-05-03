mkdir -p logs

# CUDA_VISIBLE_DEVICES=2 python train_stage25_feature_only_model5.py


# mkdir -p logs

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python train_stage25_feature_only_model5.py' \
  > logs/train_model5_gpu2.log 2>&1 < /dev/null &