from laq_model import LatentActionQuantizationStage25
from laq_model import LAQStage25Trainer
from laq_model import Stage25Dataset


jsonl_path = "/home/linhkastner/philo/datasets/ssv2/stage25_train.jsonl"


dataset = Stage25Dataset(
    jsonl_path=jsonl_path,
    image_size=256,
    code_seq_len=4,
    repeat_depth_to_3ch=True,
    depth_scale=65535.0,
)


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
    results_folder="results_stage25",
    lr=1e-4,
    save_model_every=5000,
)

trainer.train()
