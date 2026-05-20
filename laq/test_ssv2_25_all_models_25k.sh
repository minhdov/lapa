mkdir -p logs

CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model1.py \
  --checkpoint results/results_stage25_final/stage25.25000.pt \
  --z_rgb_path /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0.jsonl \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_1_val_libero10_25k/z_depth_val \
  --feature_prefix z_depth_val_model1 \
  --batch_size 256 \
  --num_workers 4

CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model2.py \
  --checkpoint results/results_stage25_final_feature_model2_ssv2_libero_ssv2/stage25.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model2_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_25k/z_depth_val \
  --feature_prefix z_depth_val_model2 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model3.py \
  --checkpoint results/results_stage252_rgbfeature_only3/stage252.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_dir /datasets/ssv2_libero_90/stage25_model_3_val_libero10_25k/z_depth_val \
  --output_prefix z_depth_val_model3 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model3_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4


  CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model4.py \
  --checkpoint results/results_model4_depth_rgb_to_zdepth_feature_libero_ssv2/model4.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_4_val_libero10_25k/z_depth_val \
  --feature_prefix z_depth_val_model4 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model4_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_gt --save_rgb

CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model5.py \
  --checkpoint results/results_model5_rgb_to_zdepth_feature/model5.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0_stage1_manifest.json \
  --output_dir /datasets/ssv2_libero_90/stage25_model_5_val_libero10_25k/z_depth_val \
  --output_prefix z_depth_val_model5 \
  --batch_size 256 \
  --num_workers 4 \
  --prefetch_factor 4 \
  --compute_metrics --save_rgb --save_gt


CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model6_1_feature.py \
  --checkpoint results/results_stage25_model2_zrgb_ssv2_libero90_to_zdepth_indices_corl_da/stage25.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model6_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_6_1_val_libero10_25k/z_depth_val \
  --feature_prefix z_depth_val_model6_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=3 python test_ssv2_25_model7_1_feature.py \
  --checkpoint results/results_stage25_model4_zrgb_ssv2_libero90_to_zdepth_feature_corl_da/model4.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json  \
  --z_depth_path /datasets/ssv2_libero_90/shards_depth_val_libero10/depth_val_shard0.jsonl \
  --z_depth_feature_manifest /datasets/ssv2_libero_90/stage1_depth_val_libero10/features/z_depth_val_shard0/z_depth_val_shard0_stage1_manifest.json \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_7_1_val_libero10_25k/z_depth_val \
  --feature_prefix z_depth_val_model7_1 \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model7_1_debug.jsonl \
  --batch_size 256 \
  --num_workers 4 --save_gt --save_rgb




