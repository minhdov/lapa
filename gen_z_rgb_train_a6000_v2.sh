
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 \
python3 -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/something-something-v2/nips/30k \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/something-something-v2/nips/30k/shards/train_folders_shard4.txt \
  --unshuffled_jsonl /datasets/something-something-v2/nips/30k/z_rgb_train_shard4_0.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/streaming_params_22485_ssv2
 
CUDA_VISIBLE_DEVICES=0 \
python3 -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/something-something-v2/nips/30k \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/something-something-v2/nips/30k/shards/train_folders_shard5.txt \
  --unshuffled_jsonl /datasets/something-something-v2/nips/30k/z_rgb_train_shard5_0.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa-depth/stage2-rgb/lapa_checkpoints/streaming_params_22485_ssv2
 
