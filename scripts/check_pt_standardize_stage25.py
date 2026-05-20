from pathlib import Path
import torch

paths = [
    "/datasets/ssv2_libero_90/stage25_model_1_val_libero10_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_2_val_libero_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_3_val_libero10_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_4_val_libero10_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_5_val_libero10_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_6_1_val_libero10_standardized/z_depth_val",
    "/datasets/ssv2_libero_90/stage25_model_7_1_val_libero10_standardized/z_depth_val",
]

for p in paths:
    d = Path(p)
    print("\nDIR:", d)

    pts = sorted(d.glob("*.pt"))
    print("num pt:", len(pts))

    if not pts:
        continue

    x = torch.load(pts[0], map_location="cpu")
    print("file:", pts[0].name)
    print("keys:", list(x.keys()))

    for k, v in x.items():
        if torch.is_tensor(v):
            print(" ", k, v.shape, v.dtype)
        elif isinstance(v, list):
            print(" ", k, "list", len(v))
        else:
            print(" ", k, type(v), v)
