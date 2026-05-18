
mkdir -p logs
python train_model_1_4_rgb.py

# nohup bash -lc 'python train_model_1_4_rgb.py' \
#   > logs/train_model_1_4_rgb.log 2>&1 < /dev/null &