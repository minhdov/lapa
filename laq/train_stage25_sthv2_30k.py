from laq_model import LatentActionQuantizationStage25
from laq_model import LAQStage25Trainer
from laq_model import Stage25Dataset
from torchvision.utils import save_image, make_grid
import torch, json

z_rgb_path ="/datasets/ssv2/nips/z_rgb_train.jsonl"
z_depth_path ="/datasets/ssv2/nips/z_depth_train.jsonl"

dataset = Stage25Dataset(
    z_rgb_path=z_rgb_path,
    z_depth_path=z_depth_path,
    image_size=256,
    code_seq_len=4,
    repeat_depth_to_3ch=True,
    depth_scale=65535.0,
)

sample = dataset[0]
print(sample["id"])
print(sample["depth1"].shape, sample["depth1"].min(), sample["depth1"].max())
print(sample["z_rgb_indices"])
print(sample["z_depth_indices"])
print(sample["depth1_path"])

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
    batch_size=64,
    grad_accum_every=1,
    num_train_steps=30000,
    results_folder="results_stage25_final",
    lr=1e-4,
    save_model_every=5000,
    log_every=50,

)

trainer.train()
