import torch
from pathlib import Path
feature_dir = Path("/datasets/libero3/features_depth_stage1/z_depth_train_shard0_with_actions")

feature_dir = Path("/datasets/ssv2_libero_90/stage25_model_1_val_libero10/z_depth_val")
files = sorted(feature_dir.glob("*.pt"))

print("num pt files:", len(files))
print("first files:", [p.name for p in files[:5]])

if files:
    obj = torch.load(files[0], map_location="cpu")
    print("keys:", obj.keys())
    for k, v in obj.items():
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v), len(v) if hasattr(v, "__len__") else v)