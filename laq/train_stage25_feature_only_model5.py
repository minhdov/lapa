from laq_model import LatentActionQuantizationStage25Model5
from laq_model import LAQStage25TrainerModel5
from laq_model import Stage252DatasetModel5

import torch


z_rgb_feature_manifest = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json"
z_depth_feature_manifest = "/datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json"


dataset = Stage252DatasetModel5(
    z_rgb_feature_manifest=z_rgb_feature_manifest,
    z_depth_feature_manifest=z_depth_feature_manifest,
    check_length_alignment=True,
    keep_z_rgb_indices=False,
)

sample = dataset[0]
print("id:", sample["id"])
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_feature:", sample["z_depth_feature"].shape, sample["z_depth_feature"].dtype)


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


laq = LatentActionQuantizationStage25Model5(
    dim=1024,
    code_seq_len=4,
    z_rgb_feature_dim=4096,
    z_depth_feature_dim=z_depth_feature_dim,
    predict_token_features=predict_token_features,
    feature_loss_weight=1.0,
    cosine_loss_weight=0.1,

    # MLP encoder config
    hidden_mult=2,
    num_mlp_layers=3,
    z_rgb_feature_dropout=0.0,
).cuda()


trainer = LAQStage25TrainerModel5(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70001,
    results_folder="results_model5_rgb_to_zdepth_feature",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,

    # Model 5 has no depth PNG loading, so it should be faster than Model 4.
    # Start with moderate workers. Increase to 16 if GPU is still under-utilized.
    num_workers=16,
    pin_memory=False,
    wandb_project="lapa_depth_model5",
    wandb_run_name="model5_rgb_to_zdepth_feature",
    save_best=True,
    save_best_every=100,
    best_metric="loss",
)


# Optional: resume from an existing Model 5 checkpoint.
# Uncomment only if the checkpoint is from LatentActionQuantizationStage25Model5.
#
# trainer.load(
#     "results_model5_rgb_to_zdepth_feature/model5.10000.pt",
#     strict=True,
# )

trainer.train()
