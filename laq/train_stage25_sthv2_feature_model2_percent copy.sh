# python train_stage25_sthv2_feature_model2_percent.py --data_percent 5
# python train_stage25_sthv2_feature_model2_percent.py --data_percent 10
CUDA_VISIBLE_DEVICES=1 python train_stage25_sthv2_feature_model2_percent.py --data_percent 20
CUDA_VISIBLE_DEVICES=1 python train_stage25_sthv2_feature_model2_percent.py --data_percent 40
CUDA_VISIBLE_DEVICES=1 python train_stage25_sthv2_feature_model2_percent.py --data_percent 80


nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 python train_stage25_sthv2_feature_model2_percent.py  --data_percent 20' \
  > /outputs/train_stage25_sthv2_feature_model2_percent_20.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 python train_stage25_sthv2_feature_model2_percent.py  --data_percent 40' \
  > /outputs/train_stage25_sthv2_feature_model2_percent_40.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=8 python train_stage25_sthv2_feature_model2_percent.py  --data_percent 60' \
  > /outputs/train_stage25_sthv2_feature_model2_percent_80.log 2>&1 < /dev/null &
