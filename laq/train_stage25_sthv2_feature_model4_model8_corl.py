
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


z_rgb_feature_manifest = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json"
z_depth_feature_manifest = "/datasets/ssv2/nips/model_1_5_rgb_depth_ssv2/features/z_depth_train_model_1_5_rgb_depth_manifest.json"
z_depth_path = "/datasets/ssv2/nips/z_depth_train.jsonl"

# z_rgb_feature_manifest = "/datasets/ssv2_libero_stage25_model4/z_rgb_train_mixed_manifest.json"
# z_depth_feature_manifest = "/datasets/ssv2_libero_stage25_model4/z_depth_train_mixed_manifest.json"
# z_depth_path = "/datasets/ssv2_libero_stage25_model4/z_depth_train_mixed.jsonl"


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

sample = dataset[0]
print("id:", sample["id"])
print("depth1:", sample["depth1"].shape, sample["depth1"].min(), sample["depth1"].max())
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_feature:", sample["z_depth_feature"].shape, sample["z_depth_feature"].dtype)
print("depth1_path:", sample["depth1_path"])

save_image(sample["depth1"][:1], "debug_depth1_model4.png", normalize=True)


# IMPORTANT:
# Set z_depth_feature_dim according to the actual shape of sample["z_depth_feature"].
# If sample["z_depth_feature"].shape == [32], use z_depth_feature_dim=32.
# If sample["z_depth_feature"].shape == [1024], use z_depth_feature_dim=1024.
# If sample["z_depth_feature"].shape == [4, 32], use z_depth_feature_dim=32 and predict_token_features=True.
# If sample["z_depth_feature"].shape == [4, 1024], use z_depth_feature_dim=1024 and predict_token_features=True.

z_depth_shape = sample["z_depth_feature"].shape

if len(z_depth_shape) == 1:
    z_depth_feature_dim = z_depth_shape[0]
    predict_token_features = False
elif len(z_depth_shape) == 2:
    z_depth_feature_dim = z_depth_shape[1]
    predict_token_features = True
else:
    raise RuntimeError(f"Unsupported z_depth_feature shape: {z_depth_shape}")

print("z_depth_feature_dim:", z_depth_feature_dim)
print("predict_token_features:", predict_token_features)


laq = LatentActionQuantizationStage25Model4(
    dim=1024,
    image_size=256,
    patch_size=32,
    spatial_depth=8,
    dim_head=64,
    heads=16,
    code_seq_len=4,
    z_rgb_feature_dim=4096,
    z_depth_feature_dim=z_depth_feature_dim,
    predict_token_features=predict_token_features,
    feature_loss_weight=1.0,
    cosine_loss_weight=0.1,
).cuda()



trainer = LAQStage25TrainerModel4(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70001,
    results_folder="results_model4_model8_depth_rgb_to_zdepth_feature_ssv2",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,
    num_workers=16,
    pin_memory=False,
    wandb_project="lapa_depth_model4",
    wandb_run_name="results_model4_model8_depth_rgb_to_zdepth_feature_ssv2",
    save_best=True,
    best_metric="loss",
)

# Optional: resume from an existing Model 4 checkpoint.
# Do NOT load old stage25 checkpoint trained for z_depth_indices with strict=True.
# Uncomment only if the checkpoint is from LatentActionQuantizationStage25Model4.
#
# trainer.load(
#     "results_model4_depth_rgb_to_zdepth_feature/model4.10000.pt",
#     strict=True,
# )

trainer.train()
