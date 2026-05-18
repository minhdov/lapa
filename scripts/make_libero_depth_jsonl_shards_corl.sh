#!/bin/bash
set -e

python make_libero_depth_jsonl_shards_corl.py \
  --depth_root /datasets/ssv2_libero_90/depth_val \
  --out_dir /datasets/ssv2_libero_90 \
  --num_shards 1 \
  --prefix depth_val

