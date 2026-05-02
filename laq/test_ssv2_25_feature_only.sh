# CUDA_VISIBLE_DEVICES=1 python inference_stage252_feature_only.py ...

mkdir -p logs

CUDA_VISIBLE_DEVICES=1 python inference_stage25_feature_only.py \
  --checkpoint results_stage252_rgbfeature_only/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
  --z_depth_path /datasets/ssv2/nips/z_depth_train.jsonl \
  --output_jsonl /datasets/ssv2/nips/stage252_predictions_train.jsonl \
  --batch_size 64 \
  --num_workers 0 \
  --max_batches 5