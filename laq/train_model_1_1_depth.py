from laq_model.latent_action_quantization import LatentActionQuantization
from laq_model.laq_trainer import LAQTrainer
import torch


# =========================
# Model 1.1-depth
# Purpose:
# Train depth-only LAQ tokenizer on 30k SSV2.
#
# Input:
#   depth1, depth2
#
# Decoder input:
#   z_depth_feature / z_depth_tokens, depth1
#
# Decoder output:
#   depth2
#
# Output used by:
#   depth-only baseline
#   z_depth_indices / z_depth_feature extraction for later stages
# =========================

# For depth-only training, both folder and depth_folder should point to depth data
# depending on how LAQTrainer/ImageVideoDataset is implemented.
depth_path = "/datasets/ssv2_libero90/depth_train"

laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=8,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    temporal_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
).cuda()


trainer = LAQTrainer(
    laq,
    folder=depth_path,
    depth_folder=depth_path,
    offsets=30,
    batch_size=64,
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=25001,
    results_folder="results/model_1_1_depth_ssv2_25k_test",
    lr=1e-4,
    save_model_every=5000,
    save_results_every=500,
)


trainer.train()