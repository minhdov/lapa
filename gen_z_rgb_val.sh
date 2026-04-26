

mkdir -p logs
nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=3 \
  python -m latent_pretraining.inference_updated_jsonl \
    --dataset_root /home/linhkastner/philo/datasets/ssv2 \
    --frames_dirname frames_val \
    --labels_dirname labels \
    --unshuffled_jsonl /home/linhkastner/philo/datasets/ssv2/z_rgb_val.jsonl \
    --mesh_dim 1,1,1,1' > logs/lapa_inference_rgb_val.log 2>&1 < /dev/null &


