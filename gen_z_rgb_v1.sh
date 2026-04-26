CUDA_VISIBLE_DEVICES=3  python -m latent_pretraining.inference \
  --dataset_root /home/linhkastner/philo/datasets/ssv2 \
  --frames_dirname frames_val \
  --labels_dirname labels \
  --output_dirname z_rgb_indices_stage2_val \
  --debug_dirname z_rgb_indices_stage2_val_debug \
  --save_debug_json
