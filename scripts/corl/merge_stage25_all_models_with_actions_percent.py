import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from tqdm import tqdm


# ============================================================
# Utils
# ============================================================

def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    data = {}

    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading JSONL {path.name}"):
            if not line.strip():
                continue

            obj = json.loads(line)
            sid = str(obj["id"])

            if sid in data:
                raise ValueError(f"Duplicate id in {path}: {sid}")

            data[sid] = obj

    return data


def parse_video_id_from_path(path: str) -> str:
    return Path(path).parent.name


def parse_video_id_from_id(sample_id: str) -> str:
    # If id is like:
    # libero_10_..._demo_demo_0_000000
    # or video_id/frame style
    # safer to remove final frame suffix if it exists.
    parts = sample_id.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    return sample_id


def load_model_parts(model_name: str, model_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all .pt parts from one model directory and return:
      id -> field dict

    Expected standardized keys inside each .pt:
      id
      depth1_path
      z_rgb_indices_input
      z_rgb_feature_input
      z_depth_indices_gt
      z_depth_indices_pred
      z_depth_feature_gt
      z_depth_feature_pred
      confidence

    Output keys will be renamed using model_name, for example:
      z_depth_feature_pred_model_5
      z_depth_indices_pred_model_5
      confidence_model_5
    """

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    pt_files = sorted(model_dir.glob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files in: {model_dir}")

    out = {}

    for pt_path in tqdm(pt_files, desc=f"Loading {model_name}"):
        pkg = torch.load(pt_path, map_location="cpu")

        ids = [str(x) for x in pkg["id"]]
        n = len(ids)

        # Sanity check first dimension
        for k, v in pkg.items():
            if torch.is_tensor(v) and v.shape[0] != n:
                raise ValueError(
                    f"{model_name} {pt_path.name} key={k} has first dim {v.shape[0]} "
                    f"but len(ids)={n}"
                )

        for i, sid in enumerate(ids):
            if sid in out:
                raise ValueError(f"Duplicate id in {model_name}: {sid}")

            item = {}

            # Metadata
            if "depth1_path" in pkg:
                item["depth1_path"] = str(pkg["depth1_path"][i])

            # Shared input / GT fields
            if "z_rgb_indices_input" in pkg:
                item["z_rgb_indices_input"] = pkg["z_rgb_indices_input"][i]

            if "z_rgb_feature_input" in pkg:
                item["z_rgb_feature_input"] = pkg["z_rgb_feature_input"][i]

            if "z_depth_indices_gt" in pkg:
                item["z_depth_indices_gt"] = pkg["z_depth_indices_gt"][i]

            if "z_depth_feature_gt" in pkg:
                item["z_depth_feature_gt"] = pkg["z_depth_feature_gt"][i]

            # Model-specific outputs
            if "z_depth_feature_pred" in pkg:
                item[f"z_depth_feature_pred_{model_name}"] = pkg["z_depth_feature_pred"][i]

            if "z_depth_indices_pred" in pkg:
                item[f"z_depth_indices_pred_{model_name}"] = pkg["z_depth_indices_pred"][i]

            if "confidence" in pkg:
                item[f"confidence_{model_name}"] = pkg["confidence"][i]

            out[sid] = item

    return out


def stack_optional_tensors(items: List[Dict[str, Any]], key: str):
    vals = [x.get(key, None) for x in items]

    if all(v is None for v in vals):
        return None

    if any(v is None for v in vals):
        missing = sum(v is None for v in vals)
        raise ValueError(f"Key {key} missing in {missing}/{len(vals)} samples")

    return torch.stack(vals, dim=0)


def get_first_existing(sample: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def infer_image_path(base_item: Dict[str, Any], action_item: Optional[Dict[str, Any]] = None):
    if action_item is not None:
        p = get_first_existing(action_item, ["image", "depth1_path", "depth_path", "rgb_path"])
        if p is not None:
            return p

    p = get_first_existing(base_item, ["image", "depth1_path", "depth_path", "rgb_path"])
    return p


def make_depth_pair_from_image(image_path: str):
    """
    If image_path is 000123.png, create [000123.png, 000124.png].
    If next path cannot be inferred, return [image_path, None].
    """
    if image_path is None:
        return None

    p = Path(image_path)
    stem = p.stem

    if not stem.isdigit():
        return [image_path, None]

    next_name = f"{int(stem) + 1:0{len(stem)}d}{p.suffix}"
    next_path = str(p.with_name(next_name))

    return [image_path, next_path]


# ============================================================
# Main merge
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge Stage 2.5 features from percent models "
            "model_5, model_10, model_20, model_40, model_60, model_80 "
            "with LIBERO action_vector and magnitude."
        )
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged .pt parts and manifest.",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="libero10_val_stage25_percent_models",
    )

    parser.add_argument(
        "--part_size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--actions_jsonl",
        type=str,
        required=True,
        help="JSONL produced by extract_libero10_actions_xyz.py.",
    )

    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="Also write lightweight JSONL metadata. Does not save large features into JSONL.",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
    )

    # Model dirs: from top to bottom
    parser.add_argument("--model_5_dir", type=str, required=True)
    parser.add_argument("--model_10_dir", type=str, required=True)
    parser.add_argument("--model_20_dir", type=str, required=True)
    parser.add_argument("--model_40_dir", type=str, required=True)
    parser.add_argument("--model_60_dir", type=str, required=True)
    parser.add_argument("--model_80_dir", type=str, required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = {
        "model_5": Path(args.model_5_dir),
        "model_10": Path(args.model_10_dir),
        "model_20": Path(args.model_20_dir),
        "model_40": Path(args.model_40_dir),
        "model_60": Path(args.model_60_dir),
        "model_80": Path(args.model_80_dir),
    }

    # ------------------------------------------------------------
    # Load actions
    # ------------------------------------------------------------

    actions_by_id = load_jsonl_by_id(Path(args.actions_jsonl))

    # ------------------------------------------------------------
    # Load model features
    # ------------------------------------------------------------

    model_data = {}
    for model_name, model_dir in model_dirs.items():
        model_data[model_name] = load_model_parts(model_name, model_dir)

    # ------------------------------------------------------------
    # Check ID alignment
    # ------------------------------------------------------------

    ref_name = "model_5"
    ref_ids = list(model_data[ref_name].keys())

    print("\nChecking id alignment...")

    for name, data in model_data.items():
        ids = list(data.keys())

        same_len = len(ids) == len(ref_ids)
        same_order = ids == ref_ids
        same_set = set(ids) == set(ref_ids)

        print(name, "same_len:", same_len, "same_order:", same_order, "same_set:", same_set)

        if not same_set:
            missing = list(set(ref_ids) - set(ids))[:5]
            extra = list(set(ids) - set(ref_ids))[:5]
            raise ValueError(f"{name} ID set mismatch. missing={missing}, extra={extra}")

    # We will use ref_ids order.
    all_ids = ref_ids

    # ------------------------------------------------------------
    # Prepare optional JSONL
    # ------------------------------------------------------------

    jsonl_f = None
    output_jsonl_path = None

    if args.write_jsonl:
        if args.output_jsonl:
            output_jsonl_path = Path(args.output_jsonl)
        else:
            output_jsonl_path = output_dir / f"{args.output_prefix}.jsonl"

        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = output_jsonl_path.open("w", encoding="utf-8")

    # ------------------------------------------------------------
    # Merge and save parts
    # ------------------------------------------------------------

    part_infos = []
    part_idx = 0
    total_saved = 0

    buffer = []

    def flush(force=False):
        nonlocal part_idx, total_saved, buffer

        if len(buffer) == 0:
            return

        if not force and len(buffer) < args.part_size:
            return

        n = len(buffer)
        out_path = output_dir / f"{args.output_prefix}_part{part_idx:05d}.pt"

        ids = [x["id"] for x in buffer]
        video_ids = [x["video_id"] for x in buffer]
        images = [x["image"] for x in buffer]
        depth_pairs = [x["depth_pair"] for x in buffer]

        action_vector = torch.tensor(
            [x["action_vector"] for x in buffer],
            dtype=torch.float32,
        )

        magnitude = torch.tensor(
            [x["magnitude"] for x in buffer],
            dtype=torch.float32,
        )

        pkg = {
            "id": ids,
            "video_id": video_ids,
            "image": images,
            "depth_pair": depth_pairs,

            "action_vector": action_vector,
            "magnitude": magnitude,
            "magnitude_mode": "xyz",
            "magnitude_definition": "sqrt(dx^2 + dy^2 + dz^2)",
        }

        # Shared inputs / GT
        shared_keys = [
            "z_rgb_indices_input",
            "z_rgb_feature_input",
            "z_depth_indices_gt",
            "z_depth_feature_gt",
        ]

        for key in shared_keys:
            tensor = stack_optional_tensors(buffer, key)
            if tensor is not None:
                pkg[key] = tensor

        # Model-specific outputs
        model_specific_keys = [
            "z_depth_feature_pred_model_5",
            "z_depth_indices_pred_model_5",
            "confidence_model_5",

            "z_depth_feature_pred_model_10",
            "z_depth_indices_pred_model_10",
            "confidence_model_10",

            "z_depth_feature_pred_model_20",
            "z_depth_indices_pred_model_20",
            "confidence_model_20",

            "z_depth_feature_pred_model_40",
            "z_depth_indices_pred_model_40",
            "confidence_model_40",

            "z_depth_feature_pred_model_60",
            "z_depth_indices_pred_model_60",
            "confidence_model_60",

            "z_depth_feature_pred_model_80",
            "z_depth_indices_pred_model_80",
            "confidence_model_80",
        ]

        for key in model_specific_keys:
            tensor = stack_optional_tensors(buffer, key)
            if tensor is not None:
                pkg[key] = tensor

        torch.save(pkg, out_path)

        part_info = {
            "part": part_idx,
            "path": str(out_path),
            "num_samples": n,
            "action_vector_shape": list(action_vector.shape),
            "magnitude_shape": list(magnitude.shape),
        }

        for key, value in pkg.items():
            if torch.is_tensor(value):
                part_info[f"{key}_shape"] = list(value.shape)

        part_infos.append(part_info)

        print(f"Saved {out_path} | samples={n}")

        total_saved += n
        part_idx += 1
        buffer = []

    missing_action = 0

    for sid in tqdm(all_ids, desc="Merging"):
        merged = {
            "id": sid,
        }

        # Collect per-model fields
        for model_name in model_dirs.keys():
            merged.update(model_data[model_name][sid])

        action_item = actions_by_id.get(sid, None)

        if action_item is None:
            missing_action += 1
            raise KeyError(f"Missing action for id={sid}")

        image_path = infer_image_path(merged, action_item)

        if image_path is None:
            raise ValueError(f"Cannot infer image path for id={sid}")

        video_id = action_item.get("video_id") or parse_video_id_from_path(image_path)

        merged["video_id"] = video_id
        merged["image"] = image_path
        merged["depth_pair"] = action_item.get("depth_pair") or make_depth_pair_from_image(image_path)

        merged["action_vector"] = action_item["action_vector"]
        merged["magnitude"] = float(action_item["magnitude"])

        if jsonl_f is not None:
            light_item = {
                "id": merged["id"],
                "video_id": merged["video_id"],
                "image": merged["image"],
                "depth_pair": merged["depth_pair"],

                "action_vector": merged["action_vector"],
                "magnitude": merged["magnitude"],
                "magnitude_mode": "xyz",

                "has_z_rgb_indices_input": "z_rgb_indices_input" in merged,
                "has_z_rgb_feature_input": "z_rgb_feature_input" in merged,
                "has_z_depth_indices_gt": "z_depth_indices_gt" in merged,
                "has_z_depth_feature_gt": "z_depth_feature_gt" in merged,

                "has_model_5": "z_depth_feature_pred_model_5" in merged,
                "has_model_10": "z_depth_feature_pred_model_10" in merged,
                "has_model_20": "z_depth_feature_pred_model_20" in merged,
                "has_model_40": "z_depth_feature_pred_model_40" in merged,
                "has_model_60": "z_depth_feature_pred_model_60" in merged,
                "has_model_80": "z_depth_feature_pred_model_80" in merged,
            }

            jsonl_f.write(json.dumps(light_item, ensure_ascii=False) + "\n")

        buffer.append(merged)
        flush(force=False)

    flush(force=True)

    if jsonl_f is not None:
        jsonl_f.close()

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    manifest = {
        "prefix": args.output_prefix,
        "total_samples": total_saved,
        "num_parts": len(part_infos),
        "feature_output_dir": str(output_dir),
        "output_jsonl": str(output_jsonl_path) if output_jsonl_path is not None else "",

        "actions_jsonl": args.actions_jsonl,
        "magnitude_mode": "xyz",
        "magnitude_definition": "sqrt(dx^2 + dy^2 + dz^2)",

        "model_dirs": {k: str(v) for k, v in model_dirs.items()},

        "schema": {
            "metadata": [
                "id",
                "video_id",
                "image",
                "depth_pair",
            ],
            "labels": [
                "z_depth_indices_gt",
                "z_depth_feature_gt",
                "action_vector",
                "magnitude",
            ],
            "inputs": [
                "z_rgb_indices_input",
                "z_rgb_feature_input",
            ],
            "stage25_predictions": [
                "z_depth_feature_pred_model_5",
                "z_depth_indices_pred_model_5",
                "confidence_model_5",

                "z_depth_feature_pred_model_10",
                "z_depth_indices_pred_model_10",
                "confidence_model_10",

                "z_depth_feature_pred_model_20",
                "z_depth_indices_pred_model_20",
                "confidence_model_20",

                "z_depth_feature_pred_model_40",
                "z_depth_indices_pred_model_40",
                "confidence_model_40",

                "z_depth_feature_pred_model_60",
                "z_depth_indices_pred_model_60",
                "confidence_model_60",

                "z_depth_feature_pred_model_80",
                "z_depth_indices_pred_model_80",
                "confidence_model_80",
            ],
        },

        "parts": part_infos,
    }

    manifest_path = output_dir / f"{args.output_prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print("output_dir:", output_dir)
    print("manifest:", manifest_path)
    print("total_saved:", total_saved)
    print("num_parts:", len(part_infos))
    print("missing_action:", missing_action)


if __name__ == "__main__":
    main()