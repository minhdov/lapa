# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs



CUDA_VISIBLE_DEVICES=2 python inference_stage25_model5.py \
  --checkpoint results_model5_rgb_to_zdepth_feature/model5.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
  --z_depth_feature_manifest /datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json \
  --output_dir /datasets/ssv2/nips/model5_predictions_train \
  --output_prefix model5_pred_z_depth_train \
  --batch_size 128 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --max_batches 5 \
  --compute_metrics