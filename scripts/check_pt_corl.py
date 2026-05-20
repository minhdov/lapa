
from pathlib import Path
import torch
import json

ROOT = Path("/datasets/ssv2_libero_90")

MODEL_DIRS = [
    "stage25_model_1_val_libero10/z_depth_val",
    "stage25_model_2_val_libero/z_depth_val",
    "stage25_model_3_val_libero10/z_depth_val",
    "stage25_model_4_val_libero10/z_depth_val",
    "stage25_model_5_val_libero10/z_depth_val",
    "stage25_model_6_1_val_libero10/z_depth_val",
    "stage25_model_7_1_val_libero10/z_depth_val",
]

def describe_value(v):
    if torch.is_tensor(v):
        return f"Tensor shape={tuple(v.shape)} dtype={v.dtype}"
    if isinstance(v, list):
        if len(v) == 0:
            return "list len=0"
        return f"list len={len(v)} first_type={type(v[0]).__name__}"
    if isinstance(v, dict):
        return f"dict keys={list(v.keys())[:10]}"
    return type(v).__name__

for model_dir in MODEL_DIRS:
    d = ROOT / model_dir
    print("\n" + "=" * 100)
    print("DIR:", d)

    if not d.exists():
        print("MISSING DIR")
        continue

    pt_files = sorted(d.glob("*.pt"))
    manifest_files = sorted(d.glob("*manifest*.json"))

    print("num .pt:", len(pt_files))
    print("manifest:", manifest_files[0] if manifest_files else "None")

    if manifest_files:
        try:
            with manifest_files[0].open("r") as f:
                m = json.load(f)
            print("manifest keys:", list(m.keys()))
            print("manifest total_samples:", m.get("total_samples"))
            print("manifest num_parts:", m.get("num_parts"))
            if "parts" in m and len(m["parts"]) > 0:
                print("first part keys:", list(m["parts"][0].keys()))
                print("first part path:", m["parts"][0].get("path"))
        except Exception as e:
            print("manifest read error:", e)

    if not pt_files:
        print("NO PT FILES")
        continue

    pt = pt_files[0]
    print("sample pt:", pt.name)

    try:
        x = torch.load(pt, map_location="cpu")
    except Exception as e:
        print("torch.load error:", e)
        continue

    print("pt keys:", list(x.keys()))

    for k, v in x.items():
        print(f"  {k}: {describe_value(v)}")
