
mkdir -p logs
# python train_model_1_5_rgb_depth.py

nohup bash -lc 'python train_model_1_5_rgb_depth.py' \
  > logs/train_model_1_5_rgb_depth.log 2>&1 < /dev/null &