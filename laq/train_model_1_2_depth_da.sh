
mkdir -p logs
python train_model_1_2_depth_da.py

# nohup bash -lc 'python train_model_1_2_depth_da.py' \
#   > logs/train_model_1_2_depth_da.log 2>&1 < /dev/null &