mkdir -p logs

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.0.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_0k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.5000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_5k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 


CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.15000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_15k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.25000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_25k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.35000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_35k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.45000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_45k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 

CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.55000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_55k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 


CUDA_VISIBLE_DEVICES=1 python test_ssv2_25_model8_1_ssv2_only.py \
  --checkpoint results_stage25_final_feature_model2_model8_ssv2/stage25.65000.pt \
  --z_rgb_feature_manifest /datasets/ssv2_libero_90/stage2_rgb_libero10/z_rgb_train_shard0_0_manifest.json \
  --z_depth_path /datasets/ssv2_libero_90/stage1_depth_val_libero10/z_depth_val_shard0.jsonl \
  --output_jsonl /datasets/ssv2_libero_90/output/stage25_predictions_val_model8_1_debug.jsonl \
  --feature_output_dir /datasets/ssv2_libero_90/stage25_model_8_1_val_libero10_65k_ssv2/z_depth_val \
  --feature_prefix z_depth_val_model8_1 \
  --batch_size 256 \
  --num_workers 4 
