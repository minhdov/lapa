from pathlib import Path
import json
import torch


# ============================================================
# Root
# ============================================================

ROOT = Path("/datasets/ssv2_libero_90")


# ============================================================
# Model input/output folders
# ============================================================

MODELS = {
    "model1": {
        "input": ROOT / "stage25_model_1_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_1_val_libero10_standardized/z_depth_val",
    },
    "model2": {
        "input": ROOT / "stage25_model_2_val_libero/z_depth_val",
        "output": ROOT / "stage25_model_2_val_libero_standardized/z_depth_val",
    },
    "model3": {
        "input": ROOT / "stage25_model_3_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_3_val_libero10_standardized/z_depth_val",
    },
    "model4": {
        "input": ROOT / "stage25_model_4_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_4_val_libero10_standardized/z_depth_val",
    },
    "model5": {
        "input": ROOT / "stage25_model_5_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_5_val_libero10_standardized/z_depth_val",
    },
    "model6_1": {
        "input": ROOT / "stage25_model_6_1_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_6_1_val_libero10_standardized/z_depth_val",
    },
    "model7_1": {
        "input": ROOT / "stage25_model_7_1_val_libero10/z_depth_val",
        "output": ROOT / "stage25_model_7_1_val_libero10_standardized/z_depth_val",
    },
}


# ============================================================
# Convert old keys to standardized keys
# ============================================================

def standardize_pkg(pkg, model_name):
    out = {}

    # ----------------------------
    # Metadata
    # ----------------------------
    if "id" in pkg:
        out["id"] = pkg["id"]

    if "depth1_path" in pkg:
        out["depth1_path"] = pkg["depth1_path"]

    # ----------------------------
    # RGB input from pretrained LAPA / Stage 2
    #
    # Do NOT call this gt.
    # It is input/conditioning signal.
    # ----------------------------
    if "z_rgb_indices" in pkg:
        out["z_rgb_indices_input"] = pkg["z_rgb_indices"]

    if "z_rgb_feature" in pkg:
        out["z_rgb_feature_input"] = pkg["z_rgb_feature"]

    elif "z_rgb_features" in pkg:
        out["z_rgb_feature_input"] = pkg["z_rgb_features"]

    # ----------------------------
    # Predicted / refined depth feature
    #
    # Unified key:
    #   z_depth_feature_pred
    # ----------------------------
    feature = None

    if "z_refined_feature" in pkg:
        feature = pkg["z_refined_feature"]

    elif "pred_z_refined_feature" in pkg:
        feature = pkg["pred_z_refined_feature"]

    elif "pred_z_depth_feature" in pkg:
        feature = pkg["pred_z_depth_feature"]

    elif "z_depth_feature" in pkg:
        feature = pkg["z_depth_feature"]

    if feature is not None:
        out["z_depth_feature_pred"] = feature

    # ----------------------------
    # Predicted depth indices
    #
    # Unified key:
    #   z_depth_indices_pred
    # ----------------------------
    if "pred_z_depth_indices" in pkg:
        out["z_depth_indices_pred"] = pkg["pred_z_depth_indices"]

    elif "z_depth_indices" in pkg:
        out["z_depth_indices_pred"] = pkg["z_depth_indices"]

    # ----------------------------
    # Ground-truth depth indices
    #
    # Unified key:
    #   z_depth_indices_gt
    # ----------------------------
    if "gt_z_depth_indices" in pkg:
        out["z_depth_indices_gt"] = pkg["gt_z_depth_indices"]

    elif "z_depth_gt" in pkg:
        out["z_depth_indices_gt"] = pkg["z_depth_gt"]

    # ----------------------------
    # Optional confidence
    # ----------------------------
    if "confidence" in pkg:
        out["confidence"] = pkg["confidence"]

    # ----------------------------
    # Extra metadata
    # ----------------------------
    out["model_name"] = model_name
    out["stage"] = "stage25"
    out["dataset"] = "libero10_val"

    return out


# ============================================================
# Standardize manifest part shape keys
# ============================================================

def standardize_part_shape_keys(part):
    """
    Rename old per-part shape keys to standardized names.

    Examples:
      z_rgb_indices_shape        -> z_rgb_indices_input_shape
      z_refined_feature_shape    -> z_depth_feature_pred_shape
      z_depth_gt_shape           -> z_depth_indices_gt_shape
    """

    rename_map = {
        # RGB input
        "z_rgb_indices_shape": "z_rgb_indices_input_shape",
        "z_rgb_feature_shape": "z_rgb_feature_input_shape",
        "z_rgb_features_shape": "z_rgb_feature_input_shape",

        # Depth indices GT
        "gt_z_depth_indices_shape": "z_depth_indices_gt_shape",
        "z_depth_gt_shape": "z_depth_indices_gt_shape",

        # Depth indices prediction
        "pred_z_depth_indices_shape": "z_depth_indices_pred_shape",
        "z_depth_indices_shape": "z_depth_indices_pred_shape",

        # Depth feature prediction
        "z_refined_feature_shape": "z_depth_feature_pred_shape",
        "pred_z_refined_feature_shape": "z_depth_feature_pred_shape",
        "pred_z_depth_feature_shape": "z_depth_feature_pred_shape",
        "z_depth_feature_shape": "z_depth_feature_pred_shape",
    }

    for old_key, new_key in rename_map.items():
        if old_key in part:
            part[new_key] = part.pop(old_key)

    return part


# ============================================================
# Update manifest
# ============================================================

def update_manifest(input_dir, output_dir, model_name):
    manifests = sorted(input_dir.glob("*manifest*.json"))

    if not manifests:
        print(f"[{model_name}] no manifest found")
        return

    manifest_path = manifests[0]

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["model_name"] = model_name
    manifest["stage"] = "stage25"
    manifest["dataset"] = "libero10_val"
    manifest["standardized_schema"] = True

    manifest["feature_key"] = "z_depth_feature_pred"
    manifest["feature_key_pred"] = "z_depth_feature_pred"
    manifest["indices_key_pred"] = "z_depth_indices_pred"
    manifest["indices_key_gt"] = "z_depth_indices_gt"

    manifest["rgb_indices_key_input"] = "z_rgb_indices_input"
    manifest["rgb_feature_key_input"] = "z_rgb_feature_input"

    manifest["feature_definition"] = {
        "z_rgb_indices_input": (
            "RGB latent token indices from pretrained LAPA / Stage 2, "
            "used as input/conditioning signal if available."
        ),
        "z_rgb_feature_input": (
            "RGB feature from pretrained LAPA / Stage 2, "
            "used as input/conditioning signal if available."
        ),
        "z_depth_feature_pred": (
            "Predicted/refined depth feature from this Stage 2.5 model."
        ),
        "z_depth_indices_pred": (
            "Predicted depth latent token indices from this Stage 2.5 model, if available."
        ),
        "z_depth_indices_gt": (
            "Ground-truth Stage 1 depth latent token indices, if available."
        ),
        "confidence": (
            "Confidence for predicted depth token indices, if available."
        ),
    }

    if "feature_output_dir" in manifest:
        manifest["feature_output_dir"] = str(output_dir)

    if "feature_dir" in manifest:
        manifest["feature_dir"] = str(output_dir)

    if "parts" in manifest:
        for part in manifest["parts"]:
            old_path = Path(part["path"])
            new_path = output_dir / old_path.name
            part["path"] = str(new_path)

            standardize_part_shape_keys(part)

    out_manifest = output_dir / manifest_path.name

    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{model_name}] wrote manifest: {out_manifest}")


# ============================================================
# Main
# ============================================================

def main():
    for model_name, cfg in MODELS.items():
        input_dir = cfg["input"]
        output_dir = cfg["output"]

        print("\n" + "=" * 100)
        print(f"[{model_name}] input : {input_dir}")
        print(f"[{model_name}] output: {output_dir}")

        if not input_dir.exists():
            print(f"[{model_name}] MISSING input dir, skip")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        pt_files = sorted(input_dir.glob("*.pt"))
        print(f"[{model_name}] num pt files: {len(pt_files)}")

        for pt_path in pt_files:
            pkg = torch.load(pt_path, map_location="cpu")
            new_pkg = standardize_pkg(pkg, model_name)

            out_path = output_dir / pt_path.name
            torch.save(new_pkg, out_path)

            print(f"  saved: {out_path.name}")
            print(f"    old keys: {list(pkg.keys())}")
            print(f"    new keys: {list(new_pkg.keys())}")

        update_manifest(input_dir, output_dir, model_name)

    print("\nDone.")


if __name__ == "__main__":
    main()