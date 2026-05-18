from laq_model import LatentActionQuantizationStage25
from laq_model import LAQStage25Trainer
from laq_model import Stage25Dataset

from torchvision.utils import save_image
import torch


# ============================================================
# Paths: SSV2 + LIBERO90 mixed data
# ============================================================

z_rgb_feature_manifest = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_rgb_train_mixed_manifest.json"
)

z_depth_path = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_depth_train_mixed_gt.jsonl"
)

# If using GT-depth version, use this instead:
# z_depth_path = (
#     "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
#     "z_depth_train_mixed_gt.jsonl"
# )


# ============================================================
# Dataset
# ============================================================

dataset = Stage25Dataset(
    z_depth_path=z_depth_path,
    z_rgb_feature_manifest=z_rgb_feature_manifest,
    image_size=256,
    code_seq_len=4,
    repeat_depth_to_3ch=True,
    depth_scale=65535.0,
    check_length_alignment=True,
    keep_z_rgb_indices=False,
)

print("Dataset length:", len(dataset))

sample = dataset[0]
print("\nFirst sample:")
print("id:", sample["id"])
print("depth1:", sample["depth1"].shape, sample["depth1"].min(), sample["depth1"].max())
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_indices:", sample["z_depth_indices"], sample["z_depth_indices"].shape)
print("depth1_path:", sample["depth1_path"])

last_sample = dataset[len(dataset) - 1]
print("\nLast sample:")
print("id:", last_sample["id"])
print("depth1_path:", last_sample["depth1_path"])
print("z_rgb_features:", last_sample["z_rgb_features"].shape, last_sample["z_rgb_features"].dtype)
print("z_depth_indices:", last_sample["z_depth_indices"], last_sample["z_depth_indices"].shape)

save_image(sample["depth1"][:1], "debug_depth1_model2_ssv2_libero90.png", normalize=True)


# ============================================================
# Quick dataloader sanity check
# ============================================================

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

print("\nBatch sanity check:")
for k, v in batch.items():
    if torch.is_tensor(v):
        print(k, v.shape, v.dtype)
    else:
        print(k, type(v), v[:2] if isinstance(v, list) else v)


# ============================================================
# Model 2
# Objective:
#   depth1 + z_rgb_features -> z_depth_indices
# ============================================================

laq = LatentActionQuantizationStage25(
    dim=1024,
    codebook_size=8,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
).cuda()


# ============================================================
# Trainer
# ============================================================

trainer = LAQStage25Trainer(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70001,
    results_folder="results_model2_zrgb_ssv2_libero90_to_zdepth_indices_corl_gt",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,
)

# Optional resume:
# Only use checkpoint from LatentActionQuantizationStage25 / Model 2.
# Do not load Model 4 checkpoint here.
#
# trainer.load(
#     "results_model2_zrgb_ssv2_libero90_to_zdepth_indices/stage25.10000.pt",
#     strict=True,
# )

trainer.train()