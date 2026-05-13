import torch
from pathlib import Path

feature_dir = Path("/datasets/libero2/features_merged_stage1_model2_model4_hdf5_actions/z_depth_train_shard0")
files = sorted(feature_dir.glob("*.pt"))

total = 0

for p in files:
    obj = torch.load(p, map_location="cpu")
    n = len(obj["id"])
    total += n

    print("\nFile:", p.name)
    print("samples:", n)
    for k, v in obj.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        elif isinstance(v, list):
            print(k, "list", len(v))

print("\nTotal samples:", total)