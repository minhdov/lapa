from laq_model import LatentActionQuantizationStage25
from laq_model import LAQStage25Trainer
from laq_model import Stage25Dataset
from torchvision.utils import save_image
import torch

# z_rgb_feature_manifest = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json"
# z_depth_path = "/datasets/ssv2/nips/z_depth_train.jsonl"

z_rgb_feature_manifest = "/datasets/ssv2_libero_stage25_model4/z_rgb_train_mixed_manifest.json"
z_depth_path = "/datasets/ssv2_libero_stage25_model4/z_depth_train_mixed.jsonl"

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

sample = dataset[0]
print("id:", sample["id"])
print("depth1:", sample["depth1"].shape, sample["depth1"].min(), sample["depth1"].max())
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_indices:", sample["z_depth_indices"])
print("depth1_path:", sample["depth1_path"])

save_image(sample["depth1"][:1], "debug_depth1.png", normalize=True)

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

trainer = LAQStage25Trainer(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70001,
    results_folder="results_stage25_final_feature_model2_ssv2_libero",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,
)

# trainer.load("results_stage25_final_feature/init/stage25.10000.pt", strict=True)


trainer.train()