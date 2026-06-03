from laq_model.latent_action_quantization import LatentActionQuantization
from laq_model.laq_trainer import LAQTrainer
import torch
import argparse


# =========================
# Model 1.1-depth
# Purpose:
# Train depth-only LAQ tokenizer on SSV2 / SSV2+LIBERO depth data.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_percent", type=float, default=100.0)
    parser.add_argument(
        "--depth_path",
        type=str,
        default="/datasets/ssv2_libero90/depth_train",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=25001,
    )
    return parser.parse_args()


args = parse_args()

assert 0 < args.data_percent <= 100, "data_percent must be in (0, 100]"

depth_path = args.depth_path

results_folder = f"/outputs/results/model_1_1_depth_ssv2_25k_percent_{int(args.data_percent)}"

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
    num_train_steps=args.num_train_steps,
    results_folder=results_folder,
    lr=1e-4,
    save_model_every=5000,
    save_results_every=500,

    # thêm dòng này nếu LAQTrainer đã hỗ trợ
    data_percent=args.data_percent,
)


trainer.train()