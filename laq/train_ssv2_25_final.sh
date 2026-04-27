mkdir -p logs

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 python train_stage25_sthv2.py' \
  > logs/train_stage25_gpu2.log 2>&1 < /dev/null &