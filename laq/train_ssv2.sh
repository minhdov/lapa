# CUDA_VISIBLE_DEVICES=3 accelerate launch train_sthv2.py


mkdir -p logs
nohup bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate lapa && CUDA_VISIBLE_DEVICES=3 python train_sthv2.py' > logs/lapa_stage1_depth.log 2>&1 < /dev/null &