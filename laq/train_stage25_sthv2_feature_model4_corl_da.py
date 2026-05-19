from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)

from laq_model.laq_stage25_trainer_feature_model4 import (
    LAQStage25TrainerModel4,
)

from laq_model.data_stage25_feature_model4 import (
    Stage252DatasetModel4,
)

from torchvision.utils import save_image
import torch


# ============================================================
# Paths: SSV2 + LIBERO90 mixed data
# ============================================================

z_rgb_feature_manifest = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_rgb_train_mixed_manifest.json"
)

z_depth_feature_manifest = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_depth_train_mixed_manifest.json"
)

z_depth_path = (
    "/datasets/ssv2_libero_90/stage2_z_rgb_ssv2_libero90/"
    "z_depth_train_mixed.jsonl"
)


# ============================================================
# Dataset
# ============================================================

dataset = Stage252DatasetModel4(
    z_depth_path=z_depth_path,
    z_rgb_feature_manifest=z_rgb_feature_manifest,
    z_depth_feature_manifest=z_depth_feature_manifest,
    image_size=256,
    repeat_depth_to_3ch=True,
    depth_scale=65535.0,
    check_length_alignment=True,
    keep_z_rgb_indices=False,
)

print("Dataset length:", len(dataset))

sample = dataset[0]

print("id:", sample["id"])
print("depth1:", sample["depth1"].shape, sample["depth1"].min(), sample["depth1"].max())
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_feature:", sample["z_depth_feature"].shape, sample["z_depth_feature"].dtype)
print("depth1_path:", sample["depth1_path"])

idx = len(dataset) - 1
sample = dataset[idx]

print("last id:", sample["id"])
print("last depth1_path:", sample["depth1_path"])
print("last z_rgb_features:", sample["z_rgb_features"].shape)
print("last z_depth_feature:", sample["z_depth_feature"].shape)

save_image(sample["depth1"][:1], "debug_depth1_model4_ssv2_libero90.png", normalize=True)


# ============================================================
# Infer target feature shape
# ============================================================

z_rgb_shape = sample["z_rgb_features"].shape
z_depth_shape = sample["z_depth_feature"].shape

if len(z_rgb_shape) != 1:
    raise RuntimeError(f"Expected z_rgb_features shape [D], got {z_rgb_shape}")

z_rgb_feature_dim = z_rgb_shape[0]

if len(z_depth_shape) == 1:
    z_depth_feature_dim = z_depth_shape[0]
    predict_token_features = False
elif len(z_depth_shape) == 2:
    z_depth_feature_dim = z_depth_shape[1]
    predict_token_features = True
else:
    raise RuntimeError(f"Unsupported z_depth_feature shape: {z_depth_shape}")

print("z_rgb_feature_dim:", z_rgb_feature_dim)
print("z_depth_feature_dim:", z_depth_feature_dim)
print("predict_token_features:", predict_token_features)


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
# Model 4
# Objective:
#   depth1 + z_rgb_features -> z_depth_feature
# ============================================================

laq = LatentActionQuantizationStage25Model4(
    dim=1024,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
    z_rgb_feature_dim=z_rgb_feature_dim,
    z_depth_feature_dim=z_depth_feature_dim,
    predict_token_features=predict_token_features,
    feature_loss_weight=1.0,
    cosine_loss_weight=0.1,
).cuda()


# ============================================================
# Trainer
# ============================================================

trainer = LAQStage25TrainerModel4(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70001,
    results_folder="results_model4_zrgb_ssv2_libero90_to_zdepth_feature_corl",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,
    num_workers=12,
    pin_memory=True,
    wandb_project="lapa_depth_model4",
    wandb_run_name="model4_zrgb_ssv2_libero90_to_zdepth_feature",
    save_best=True,
    best_metric="loss",
)

# Optional resume:
# Only use checkpoint from LatentActionQuantizationStage25Model4.
# Do not load Model 2 / z_depth_indices checkpoint here.
#
# trainer.load(
#     "results_model4_zrgb_ssv2_libero90_to_zdepth_feature/model4.10000.pt",
#     strict=True,
# )

trainer.train()