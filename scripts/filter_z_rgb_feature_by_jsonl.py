import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm


def load_source_jsonl(jsonl_path: Path, keep_regex: str):
    """
    Read source RGB jsonl and keep only samples matching keep_regex.
    Matching is applied to id and video_id.
    """
    pattern = re.compile(keep_regex)
    keep_items = {}

    total = 0
    kept = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            total += 1

            sample_id = str(item["id"])
            video_id = str(item.get("video_id", ""))

            if pattern.search(sample_id) or pattern.search(video_id):
                keep_items[sample_id] = item
                kept += 1

    print(f"source_jsonl: {jsonl_path}")
    print(f"jsonl total: {total}")
    print(f"jsonl kept:  {kept}")

    return keep_items


def get_pkg_field(pkg, names, default=None):
    for name in names:
        if name in pkg:
            return pkg[name]
    return default


def flush_buffer(buffer, out_dir: Path, prefix: str, part_idx: int, manifest_parts: list):
    if len(buffer["ids"]) == 0:
        return part_idx

    out_path = out_dir / f"{prefix}_part{part_idx:05d}.pt"

    z_rgb_indices = torch.stack(buffer["z_rgb_indices"], dim=0).long()
    z_rgb_features = torch.stack(buffer["z_rgb_features"], dim=0)

    pkg = {
        "ids": list(buffer["ids"]),
        "video_ids": list(buffer["video_ids"]),
        "image_paths": list(buffer["image_paths"]),
        "instructions": list(buffer["instructions"]),
        "z_rgb_indices": z_rgb_indices,
        "z_rgb_features": z_rgb_features,
    }

    torch.save(pkg, out_path)

    part_info = {
        "part": part_idx,
        "path": str(out_path),
        "num_samples": len(buffer["ids"]),
        "z_rgb_indices_shape": list(z_rgb_indices.shape),
        "z_rgb_features_shape": list(z_rgb_features.shape),
    }

    manifest_parts.append(part_info)

    print(
        f"Saved {out_path} | "
        f"samples={len(buffer['ids'])} | "
        f"indices={list(z_rgb_indices.shape)} | "
        f"features={list(z_rgb_features.shape)}"
    )

    for k in buffer:
        buffer[k].clear()

    return part_idx + 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_jsonl", type=str, required=True)
    parser.add_argument("--source_manifest", type=str, required=True)

    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--out_feature_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)

    parser.add_argument(
        "--keep_regex",
        type=str,
        default=r"^libero_90_",
        help="Regex used to keep LIBERO90 samples. It is matched against id and video_id.",
    )

    parser.add_argument("--part_size", type=int, default=8192)

    args = parser.parse_args()

    source_jsonl = Path(args.source_jsonl)
    source_manifest = Path(args.source_manifest)
    out_jsonl = Path(args.out_jsonl)
    out_feature_dir = Path(args.out_feature_dir)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_feature_dir.mkdir(parents=True, exist_ok=True)

    keep_items = load_source_jsonl(source_jsonl, args.keep_regex)
    keep_ids = set(keep_items.keys())

    if len(keep_ids) == 0:
        print("[Warn] No samples matched keep_regex.")
        print("keep_regex:", args.keep_regex)
        print("Please check sample id format:")
        print(f"  head -n 3 {source_jsonl}")
        return

    with source_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    buffer = {
        "ids": [],
        "video_ids": [],
        "image_paths": [],
        "instructions": [],
        "z_rgb_indices": [],
        "z_rgb_features": [],
    }

    manifest_parts = []
    part_idx = 0
    total_seen = 0
    total_kept = 0
    kept_ids_from_feature = set()

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for part in tqdm(manifest["parts"], desc=f"Filtering {source_manifest.parent.name}"):
            part_path = Path(part["path"])

            if not part_path.exists():
                raise FileNotFoundError(f"Missing feature part: {part_path}")

            pkg = torch.load(part_path, map_location="cpu")

            ids = get_pkg_field(pkg, ["ids", "id"])
            video_ids = get_pkg_field(pkg, ["video_ids", "video_id"])
            image_paths = get_pkg_field(pkg, ["image_paths", "image"])
            instructions = get_pkg_field(pkg, ["instructions", "instruction"], default=None)

            z_rgb_indices = pkg["z_rgb_indices"]
            z_rgb_features = pkg["z_rgb_features"]

            if ids is None:
                raise KeyError(f"No ids/id key in {part_path}. Keys: {list(pkg.keys())}")

            n = len(ids)
            total_seen += n

            for i in range(n):
                sample_id = str(ids[i])

                if sample_id not in keep_ids:
                    continue

                source_item = keep_items[sample_id]

                video_id = str(video_ids[i]) if video_ids is not None else str(source_item.get("video_id", ""))
                image_path = str(image_paths[i]) if image_paths is not None else str(source_item.get("image", ""))
                instruction = str(instructions[i]) if instructions is not None else str(source_item.get("instruction", ""))

                indices_i = z_rgb_indices[i].detach().cpu()
                feature_i = z_rgb_features[i].detach().cpu()

                buffer["ids"].append(sample_id)
                buffer["video_ids"].append(video_id)
                buffer["image_paths"].append(image_path)
                buffer["instructions"].append(instruction)
                buffer["z_rgb_indices"].append(indices_i)
                buffer["z_rgb_features"].append(feature_i)

                out_item = dict(source_item)
                out_item["id"] = sample_id
                out_item["video_id"] = video_id
                out_item["image"] = image_path
                out_item["delta"] = [str(x) for x in indices_i.tolist()]
                out_item["instruction"] = instruction
                out_item["vision"] = source_item.get("vision", [])
                out_item["fields"] = source_item.get("fields", "[instruction],[vision],delta")

                fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                total_kept += 1
                kept_ids_from_feature.add(sample_id)

                if len(buffer["ids"]) >= args.part_size:
                    part_idx = flush_buffer(
                        buffer=buffer,
                        out_dir=out_feature_dir,
                        prefix=args.prefix,
                        part_idx=part_idx,
                        manifest_parts=manifest_parts,
                    )

    part_idx = flush_buffer(
        buffer=buffer,
        out_dir=out_feature_dir,
        prefix=args.prefix,
        part_idx=part_idx,
        manifest_parts=manifest_parts,
    )

    missing_ids = sorted(list(keep_ids - kept_ids_from_feature))

    out_manifest = {
        "prefix": args.prefix,
        "total_samples": total_kept,
        "num_parts": len(manifest_parts),
        "source_jsonl": str(source_jsonl),
        "source_manifest": str(source_manifest),
        "out_jsonl": str(out_jsonl),
        "out_feature_dir": str(out_feature_dir),
        "keep_regex": args.keep_regex,
        "parts": manifest_parts,
        "missing_count": len(missing_ids),
        "missing_examples": missing_ids[:20],
    }

    out_manifest_path = out_feature_dir / f"{args.prefix}_manifest.json"

    with out_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("total_seen_feature:", total_seen)
    print("total_keep_jsonl:", len(keep_ids))
    print("total_kept_feature:", total_kept)
    print("missing_count:", len(missing_ids))
    print("out_jsonl:", out_jsonl)
    print("out_manifest:", out_manifest_path)

    if len(missing_ids) > 0:
        print("Missing examples:")
        for x in missing_ids[:20]:
            print(" ", x)


if __name__ == "__main__":
    main()