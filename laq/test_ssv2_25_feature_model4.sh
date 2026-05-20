# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs

CUDA_VISIBLE_DEVICES=2 python inference_stage25_model4.py \
  --checkpoint results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
  --z_depth_path /datasets/ssv2/nips/z_depth_train.jsonl \
  --z_rgb_feature_manifest /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
  --z_depth_feature_manifest /datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json \
  --output_dir /datasets/ssv2/nips/model4_predictions_train \
  --output_prefix model4_pred_z_depth_train \
  --batch_size 128 \
  --max_batches 5 \
  --compute_metrics