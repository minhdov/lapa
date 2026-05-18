from laq_model.latent_action_quantization_model_1_5_rgb_depth import LatentActionQuantization
from laq_model.laq_trainer_model_1_5_rgb_depth import LAQTrainer
import torch


# =========================
# Model 1.5-rgb-depth
# Purpose:
# Train RGB-guided Depth LAQ tokenizer on SSV2.
#
# Input:
#   depth1, depth2
#
# Extra condition:
#   z_rgb_tokens from Model 1.4-rgb
#
# Decoder input:
#   z_depth_feature / z_depth_tokens, z_rgb_tokens, depth1
#
# Decoder output:
#   depth2
#
# z_rgb_tokens shape:
#   [B, 4, 1024]
#
# Dataset return:
#   cat_img, cat_depth, z_rgb_tokens
#
# Output used by:
#   RGB-guided depth latent representation
#   z_depth_indices / z_depth_feature extraction for later stages
# =========================


rgb_path = "/datasets/ssv2_libero90/frames_train"
depth_path = "/datasets/ssv2_libero90/depth_train"

z_rgb_feature_manifest = (
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/"
    "features/z_rgb_train_all_model_1_4_rgb_manifest.json"
)

# NOTE:
# If training on all SSV2 samples, it is better to use a merged manifest
# containing all 4 shards instead of only shard0.
#
# Example preferred path later:
# z_rgb_feature_manifest = (
#     "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/"
#     "features/z_rgb_train_all_model_1_4_rgb_manifest.json"
# )


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
    results_folder="results/model_1_5_rgb_depth_ssv2_rgb25k_depth25k",
    lr=1e-4,
    save_model_every=5000,
    save_results_every=500,

    # Model 1.5 specific
    z_rgb_feature_manifest=z_rgb_feature_manifest,
    return_rgb_tokens=True,
    strict_rgb_token=True,
)
    

trainer.train()