mkdir -p logs

nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/ssv2/nips \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/ssv2/nips/shards/train_folders_shard1.txt \
  --unshuffled_jsonl /datasets/ssv2/nips/z_rgb_train_shard1.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa/lapa_checkpoints/streaming_params_22485_ssv2' \
  > logs/lapa_inference_rgb_train_shard1.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=1 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/ssv2/nips \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/ssv2/nips/shards/train_folders_shard2.txt \
  --unshuffled_jsonl /datasets/ssv2/nips/z_rgb_train_shard2.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa/lapa_checkpoints/streaming_params_22485_ssv2' \
  > logs/lapa_inference_rgb_train_shard2.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/ssv2/nips \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/ssv2/nips/shards/train_folders_shard4.txt \
  --unshuffled_jsonl /datasets/ssv2/nips/z_rgb_train_shard4.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa/lapa_checkpoints/streaming_params_22485_ssv2' \
  > logs/lapa_inference_rgb_train_shard4.log 2>&1 < /dev/null &

nohup bash -lc 'CUDA_VISIBLE_DEVICES=3 \
python -m latent_pretraining.inference_update_jsonl_train \
  --dataset_root /datasets/ssv2/nips \
  --frames_dirname frames_train \
  --labels_dirname labels \
  --folder_list /datasets/ssv2/nips/shards/train_folders_shard5.txt \
  --unshuffled_jsonl /datasets/ssv2/nips/z_rgb_train_shard5.jsonl \
  --mesh_dim 1,1,1,1 \
  --vqgan_checkpoint /checkpoints/lapa/lapa_checkpoints/vqgan \
  --vocab_file /checkpoints/lapa/lapa_checkpoints/tokenizer.model \
  --load_checkpoint params::/checkpoints/lapa/lapa_checkpoints/streaming_params_22485_ssv2' \
  > logs/lapa_inference_rgb_train_shard5.log 2>&1 < /dev/null &