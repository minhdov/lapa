from laq_model import LatentActionQuantizationStage252
from laq_model import LAQStage252Trainer
from laq_model import Stage252Dataset
from torchvision.utils import save_image
import torch


z_rgb_feature_manifest = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json"
z_depth_path = "/datasets/ssv2/nips/z_depth_train.jsonl"


dataset = Stage252Dataset(
    z_depth_path=z_depth_path,
    z_rgb_feature_manifest=z_rgb_feature_manifest,
    code_seq_len=4,
    check_length_alignment=True,
    keep_z_rgb_indices=False,
)

sample = dataset[0]
print("id:", sample["id"])
print("z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
print("z_depth_indices:", sample["z_depth_indices"])
print("depth1_path:", sample["depth1_path"])


laq = LatentActionQuantizationStage252(
    dim=1024,
    codebook_size=8,
    code_seq_len=4,
    z_rgb_feature_dim=4096,
).cuda()


trainer = LAQStage252Trainer(
    laq,
    dataset=dataset,
    batch_size=128,
    grad_accum_every=1,
    num_train_steps=70000,
    results_folder="results_stage252_rgbfeature_only4",
    lr=1e-4,
    save_model_every=5000,
    log_every=100,
    wandb_project="lapa_depth_stage252",
    wandb_run_name="stage252_rgbfeature_only4",
    save_best=True,
    best_metric="acc",
)

trainer.train()
