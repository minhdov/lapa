from pathlib import Path
import torch


ROOT = Path("/datasets/ssv2_libero_90")

MODEL_DIRS = {
    "model1": ROOT / "stage25_model_1_val_libero10_standardized/z_depth_val",
    "model2": ROOT / "stage25_model_2_val_libero_standardized/z_depth_val",
    "model3": ROOT / "stage25_model_3_val_libero10_standardized/z_depth_val",
    "model4": ROOT / "stage25_model_4_val_libero10_standardized/z_depth_val",
    "model5": ROOT / "stage25_model_5_val_libero10_standardized/z_depth_val",
    "model6_1": ROOT / "stage25_model_6_1_val_libero10_standardized/z_depth_val",
    "model7_1": ROOT / "stage25_model_7_1_val_libero10_standardized/z_depth_val",
}


def load_all(model_name, model_dir):
    ids = []
    depth1_paths = []

    z_depth_indices_gt = []
    z_depth_indices_pred = []
    z_rgb_indices_input = []
    z_depth_feature_pred = []

    pt_files = sorted(model_dir.glob("*.pt"))

    if not pt_files:
        raise RuntimeError(f"No .pt files found for {model_name}: {model_dir}")

    keys_seen = None

    for pt in pt_files:
        pkg = torch.load(pt, map_location="cpu")

        if keys_seen is None:
            keys_seen = list(pkg.keys())

        ids.extend(pkg["id"])

        if "depth1_path" in pkg:
            depth1_paths.extend(pkg["depth1_path"])

        if "z_depth_indices_gt" in pkg:
            z_depth_indices_gt.append(pkg["z_depth_indices_gt"])

        if "z_depth_indices_pred" in pkg:
            z_depth_indices_pred.append(pkg["z_depth_indices_pred"])

        if "z_rgb_indices_input" in pkg:
            z_rgb_indices_input.append(pkg["z_rgb_indices_input"])

        if "z_depth_feature_pred" in pkg:
            z_depth_feature_pred.append(pkg["z_depth_feature_pred"])

    out = {
        "ids": ids,
        "depth1_paths": depth1_paths if depth1_paths else None,
        "keys": keys_seen,
        "num_samples": len(ids),
    }

    if z_depth_indices_gt:
        out["z_depth_indices_gt"] = torch.cat(z_depth_indices_gt, dim=0)

    if z_depth_indices_pred:
        out["z_depth_indices_pred"] = torch.cat(z_depth_indices_pred, dim=0)

    if z_rgb_indices_input:
        out["z_rgb_indices_input"] = torch.cat(z_rgb_indices_input, dim=0)

    if z_depth_feature_pred:
        out["z_depth_feature_pred"] = torch.cat(z_depth_feature_pred, dim=0)

    return out


def compare_list(name_a, list_a, name_b, list_b, field_name):
    same_len = len(list_a) == len(list_b)
    same_order = list_a == list_b
    same_set = set(list_a) == set(list_b)

    print(f"{field_name}: {name_a} vs {name_b}")
    print(f"  same_len   : {same_len}")
    print(f"  same_order : {same_order}")
    print(f"  same_set   : {same_set}")

    if not same_order:
        n = min(len(list_a), len(list_b))
        for i in range(n):
            if list_a[i] != list_b[i]:
                print(f"  first mismatch index: {i}")
                print(f"  {name_a}: {list_a[i]}")
                print(f"  {name_b}: {list_b[i]}")
                break

    print()


def compare_tensor(name_a, tensor_a, name_b, tensor_b, field_name):
    same_shape = tuple(tensor_a.shape) == tuple(tensor_b.shape)
    print(f"{field_name}: {name_a} vs {name_b}")
    print(f"  shape {name_a}: {tuple(tensor_a.shape)}")
    print(f"  shape {name_b}: {tuple(tensor_b.shape)}")
    print(f"  same_shape: {same_shape}")

    if not same_shape:
        print()
        return

    equal = torch.equal(tensor_a, tensor_b)
    print(f"  exactly_equal: {equal}")

    if not equal:
        diff = tensor_a != tensor_b
        num_diff = diff.sum().item()
        total = tensor_a.numel()
        print(f"  num_diff: {num_diff}/{total} = {num_diff / max(total, 1):.6f}")

        idx = diff.nonzero()
        if idx.numel() > 0:
            first = idx[0].tolist()
            print(f"  first mismatch index: {first}")
            print(f"  {name_a}: {tensor_a[tuple(first)].item()}")
            print(f"  {name_b}: {tensor_b[tuple(first)].item()}")

    print()


def token_acc(pred, gt):
    if tuple(pred.shape) != tuple(gt.shape):
        return None

    correct = (pred == gt).sum().item()
    total = gt.numel()
    return correct / max(total, 1)


def main():
    data = {}

    print("Loading models...")
    for model_name, model_dir in MODEL_DIRS.items():
        if not model_dir.exists():
            print(f"[SKIP] missing: {model_name} {model_dir}")
            continue

        d = load_all(model_name, model_dir)
        data[model_name] = d

        print(f"\n{model_name}")
        print(f"  dir        : {model_dir}")
        print(f"  num_samples: {d['num_samples']}")
        print(f"  keys       : {d['keys']}")

        if "z_depth_feature_pred" in d:
            print(f"  z_depth_feature_pred: {tuple(d['z_depth_feature_pred'].shape)}")

        if "z_depth_indices_pred" in d:
            print(f"  z_depth_indices_pred: {tuple(d['z_depth_indices_pred'].shape)}")

        if "z_depth_indices_gt" in d:
            print(f"  z_depth_indices_gt  : {tuple(d['z_depth_indices_gt'].shape)}")

        if "z_rgb_indices_input" in d:
            print(f"  z_rgb_indices_input : {tuple(d['z_rgb_indices_input'].shape)}")

        if d["depth1_paths"] is not None:
            print(f"  depth1_path count   : {len(d['depth1_paths'])}")

    print("\n" + "=" * 100)
    print("1. Check ID alignment against model1")
    print("=" * 100)

    ref_name = "model1"
    ref_ids = data[ref_name]["ids"]

    for name, d in data.items():
        compare_list(ref_name, ref_ids, name, d["ids"], "id")

    print("\n" + "=" * 100)
    print("2. Check depth1_path alignment against model1, if available")
    print("=" * 100)

    ref_depth_paths = data[ref_name]["depth1_paths"]

    for name, d in data.items():
        if ref_depth_paths is None or d["depth1_paths"] is None:
            print(f"depth1_path: skip {ref_name} vs {name}, missing depth1_path")
            continue

        compare_list(ref_name, ref_depth_paths, name, d["depth1_paths"], "depth1_path")

    print("\n" + "=" * 100)
    print("3. Check z_depth_indices_gt consistency across models")
    print("=" * 100)

    gt_models = [name for name, d in data.items() if "z_depth_indices_gt" in d]

    if not gt_models:
        print("No models have z_depth_indices_gt")
    else:
        gt_ref_name = gt_models[0]
        gt_ref = data[gt_ref_name]["z_depth_indices_gt"]

        for name in gt_models:
            compare_tensor(
                gt_ref_name,
                gt_ref,
                name,
                data[name]["z_depth_indices_gt"],
                "z_depth_indices_gt",
            )

    print("\n" + "=" * 100)
    print("4. Check z_rgb_indices_input consistency across models")
    print("=" * 100)

    rgb_models = [name for name, d in data.items() if "z_rgb_indices_input" in d]

    if len(rgb_models) <= 1:
        print("Only one or zero models have z_rgb_indices_input, skip cross-model comparison.")
    else:
        rgb_ref_name = rgb_models[0]
        rgb_ref = data[rgb_ref_name]["z_rgb_indices_input"]

        for name in rgb_models:
            compare_tensor(
                rgb_ref_name,
                rgb_ref,
                name,
                data[name]["z_rgb_indices_input"],
                "z_rgb_indices_input",
            )

    print("\n" + "=" * 100)
    print("5. Token accuracy: z_depth_indices_pred vs z_depth_indices_gt")
    print("=" * 100)

    for name, d in data.items():
        if "z_depth_indices_pred" not in d or "z_depth_indices_gt" not in d:
            print(f"{name}: skip, missing pred or gt indices")
            continue

        acc = token_acc(d["z_depth_indices_pred"], d["z_depth_indices_gt"])
        if acc is None:
            print(
                f"{name}: shape mismatch "
                f"pred={tuple(d['z_depth_indices_pred'].shape)} "
                f"gt={tuple(d['z_depth_indices_gt'].shape)}"
            )
        else:
            print(f"{name}: token_acc = {acc:.6f}")

    print("\n" + "=" * 100)
    print("6. Feature shape check")
    print("=" * 100)

    for name, d in data.items():
        if "z_depth_feature_pred" not in d:
            print(f"{name}: missing z_depth_feature_pred")
            continue

        feat = d["z_depth_feature_pred"]
        print(
            f"{name}: z_depth_feature_pred shape={tuple(feat.shape)}, "
            f"dtype={feat.dtype}, "
            f"mean_norm={feat.norm(dim=-1).mean().item():.6f}"
        )


if __name__ == "__main__":
    main()