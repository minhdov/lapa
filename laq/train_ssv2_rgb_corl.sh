# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py

# mkdir -p logs
# python train_sthv2_rgb.py

mkdir -p logs

nohup bash -lc 'python train_sthv2_rgb.py' \
  > logs/train_sthv2_rgb.log 2>&1 < /dev/null &