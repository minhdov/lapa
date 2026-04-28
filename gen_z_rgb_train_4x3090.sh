mkdir -p logs

python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/something-something-v2/nips/30k \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/something-something-v2/nips/30k/shards/train_folders_shard0.txt \
  --unshuffled_jsonl /datasets/something-something-v2/nips/30k/z_rgb_train_shard0.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/streaming_params_22485_ssv2


python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/something-something-v2/nips/30k \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/something-something-v2/nips/30k/shards/train_folders_shard3.txt \
  --unshuffled_jsonl /datasets/something-something-v2/nips/30k/z_rgb_train_shard3.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/streaming_params_22485_ssv2




# mkdir -p logs

# nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 \
# python -m latent_pretraining.inference_update_jsonl_train \
#   --dataset_root /datasets/ssv2/nips \
#   --frames_dirname frames_train \
#   --labels_dirname labels \
#   --folder_list /datasets/ssv2/nips/shards/train_folders_shard1.txt \
#   --unshuffled_jsonl /datasets/ssv2/nips/z_rgb_train_shard1.jsonl \
#   --mesh_dim 1,1,1,1 \
#   --vqgan_checkpoint /checkpoints/lapa/lapa_checkpoints/vqgan \
#   --vocab_file /checkpoints/lapa/lapa_checkpoints/tokenizer.model \
#   --load_checkpoint params::/checkpoints/lapa/lapa_checkpoints/streaming_params_22485_ssv2' \
#   > logs/lapa_inference_rgb_train_shard1.log 2>&1 < /dev/null &

