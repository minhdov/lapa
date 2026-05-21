
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python train_model_1_1_depth_percent.py --data_percent 5
CUDA_VISIBLE_DEVICES=0 python train_model_1_1_depth_percent.py --data_percent 10