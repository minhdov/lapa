from laq_model import LAQTrainer
from laq_model import LatentActionQuantization
import torch

# rgb_path = '/home/linhkastner/philo/datasets/ssv2/depth_train'
# depth_path = '/home/linhkastner/philo/datasets/ssv2/depth_train'

rgb_path = '/datasets/ssv2_libero90/frames_train'
depth_path = '/datasets/ssv2_libero90/depth_train'

laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).cuda()


trainer = LAQTrainer(
    laq,
    folder = rgb_path,
    depth_folder = depth_path,
    offsets = 30,
    batch_size = 64,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 25001,
    results_folder='results-sthv2-a6000-500-25k-rgb',
    lr=1e-4,
    save_model_every=5000,
    save_results_every=500,
)


# ckpt_path = "results-sthv2-libero90-a6000-500/vae.30000.pt"

# ckpt = torch.load(ckpt_path, map_location="cpu")

# if isinstance(ckpt, dict) and "model" in ckpt:
#     state_dict = ckpt["model"]
# else:
#     state_dict = ckpt

# laq.load_state_dict(state_dict)


# trainer.load("results-sthv2-libero90-a6000-500/vae.30000.pt")

trainer.train()        

