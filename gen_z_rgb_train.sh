

# mkdir -p logs
# nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=3 \
#   python -m latent_pretraining.inference_updated_jsonl \
#     --dataset_root /home/linhkastner/philo/datasets/ssv2 \
#     --frames_dirname frames_train \
#     --labels_dirname labels \
#     --unshuffled_jsonl /home/linhkastner/philo/datasets/ssv2/z_rgb_train.jsonl \
#     --mesh_dim 1,1,1,1' > logs/lapa_inference_rgb_train.log 2>&1 < /dev/null &

mkdir -p logs /mnt/hdd/Linh/philo/datasets/ssv2/nips

nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=2 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /home/linhkastner/philo/datasets/ssv2 \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /home/linhkastner/philo/datasets/ssv2/shards/train_folders_shard0.txt \
  --unshuffled_jsonl /mnt/hdd/Linh/philo/datasets/ssv2/nips/z_rgb_train_shard0.jsonl \
  --mesh_dim 1,1,1,1' > logs/lapa_inference_rgb_train_shard0.log 2>&1 < /dev/null &

nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=3 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /home/linhkastner/philo/datasets/ssv2 \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /home/linhkastner/philo/datasets/ssv2/shards/train_folders_shard1.txt \
  --unshuffled_jsonl /mnt/hdd/Linh/philo/datasets/ssv2/nips/z_rgb_train_shard1.jsonl \
  --mesh_dim 1,1,1,1' > logs/lapa_inference_rgb_train_shard1.log 2>&1 < /dev/null &