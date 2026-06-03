python3 merge_stage25_all_models_with_actions_percent.py \
  --actions_jsonl /datasets/ssv2_libero_90/stage0_action_val_libero10/z_depth_val_shard0_with_actions.jsonl \
  --output_dir /datasets/ssv2_libero_90/stage25_all_models_val_libero10_percent_65000 \
  --output_prefix all_models_val_libero10_percent_65000 \
  --write_jsonl \
  --model_5_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_5_percent_65000_steps/z_depth_val \
  --model_10_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_10_percent_65000_steps/z_depth_val \
  --model_20_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_20_percent_65000_steps/z_depth_val \
  --model_40_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_40_percent_65000_steps/z_depth_val \
  --model_60_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_60_percent_65000_steps/z_depth_val \
  --model_80_dir /datasets/ssv2_libero_90/stage25_model_2_val_libero10_percent/stage25_model_2_val_libero10_80_percent_65000_steps/z_depth_val


