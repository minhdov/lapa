
  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.0.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_0k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb


  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.5000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_5k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.15000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_15k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_25k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.35000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_35k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.45000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_45k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.55000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_55k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb



  CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature/model4.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_65k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb


