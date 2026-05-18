from laq_model.latent_action_quantization_model_1_4_rgb import LatentActionQuantization
from laq_model.laq_trainer_model_1_4_rgb import LAQTrainer
import torch


# =========================
# Model 1.4-rgb
# Purpose:
# Train RGB LAQ tokenizer on SSV2/LIBERO90 frames.
#
# Input:
#   rgb1, rgb2
#
# Decoder input:
#   z_rgb_feature / z_rgb_tokens, rgb1
#
# Decoder output:
#   rgb2
#
# Output used by:
#   Model 1.5-rgb-depth or Stage 1.2 RGB-guided depth
# =========================

rgb_path = "/datasets/ssv2_libero90/frames_train"
depth_path = "/datasets/ssv2_libero90/depth_train"  # kept only if trainer requires depth_folder argument

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
    folder=rgb_path,
    depth_folder=depth_path,
    offsets=30,
    batch_size=64,
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=25001,
    results_folder="results/model_1_4_rgb_ssv2_25k_test",
    lr=1e-4,
    save_model_every=5000,
    save_results_every=500,
)


trainer.train()