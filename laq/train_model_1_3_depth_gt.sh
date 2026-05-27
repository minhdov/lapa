
mkdir -p logs
python train_model_1_3_depth_gt.py

# nohup bash -lc 'python train_model_1_3_depth_gt.py' \
#   > logs/train_model_1_3_depth_gt.log 2>&1 < /dev/null &