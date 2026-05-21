from pathlib import Path
import torch

PT_DIR = Path("/datasets/ssv2_libero_90/stage25_all_models_val_libero10")

pt_files = sorted(PT_DIR.glob("*.pt"))

print("PT_DIR:", PT_DIR)
print("num pt files:", len(pt_files))

for pt in pt_files:
    print("\n" + "=" * 100)
    print("file:", pt.name)

    x = torch.load(pt, map_location="cpu")

    for k, v in x.items():
        if torch.is_tensor(v):
            print(f"{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{k}: list len={len(v)}")
        else:
            print(f"{k}: {type(v).__name__} value={v}")