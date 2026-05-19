# python make_libero_depth_jsonl_shards.py \
#   --depth_root /datasets/ssv2_libero90/frames_train \
#   --out_dir /datasets/ssv2_libero90/shards_rgb \
#   --num_shards 4 \
#   --prefix frames_train

  # python make_libero_depth_jsonl_shards.py \
  # --depth_root /datasets/ssv2_libero90/depth_train \
  # --out_dir /datasets/ssv2_libero90/shards_depth \
  # --num_shards 4 \
  # --prefix depth_train

  #   python make_libero_depth_jsonl_shards.py \
  # --depth_root /datasets/ssv2_libero_90/depth_val \
  # --out_dir /datasets/ssv2_libero_90/shards_depth_val \
  # --num_shards 1 \
  # --prefix depth_val

  python make_libero_depth_jsonl_shards.py \
  --depth_root /datasets/ssv2_libero_90/depth_train_gt \
  --out_dir /datasets/ssv2_libero_90/shards_depth_train_gt \
  --num_shards 4 \
  --prefix depth_train

