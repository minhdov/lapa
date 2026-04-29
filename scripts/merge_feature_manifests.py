#!/usr/bin/env python3
"""
merge_feature_manifests.py

Merge multiple LAPA feature manifest files into one global manifest.

Expected input structure:

features/
  z_rgb_train_shard0_0/
    z_rgb_train_shard0_0_manifest.json
    z_rgb_train_shard0_0_part00000.pt
    ...
  z_rgb_train_shard1_0/
    z_rgb_train_shard1_0_manifest.json
    ...

Output:

features/
  z_rgb_train_all_manifest.json

Usage examples:

  python merge_feature_manifests.py \
    --root /datasets/ssv2/nips/features \
    --pattern "z_rgb_train_shard*_0/*_manifest.json" \
    --output /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
    --prefix z_rgb_train_all

If your manifest files are .jsonl, use:

  python merge_feature_manifests.py \
    --root /datasets/ssv2/nips/features \
    --pattern "z_rgb_train_shard*_0/*_manifest.jsonl" \
    --output /datasets/ssv2/nips/features/z_rgb_train_all_manifest.json \
    --prefix z_rgb_train_all

Notes:
- This script merges manifest metadata only.
- It does NOT merge .pt files.
- It preserves the original .pt file paths.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_manifest(path: Path) -> Dict[str, Any]:
    """
    Load either:
    1. A normal JSON manifest:
       {
         "prefix": "...",
         "total_samples": ...,
         "num_parts": ...,
         "parts": [...]
       }

    2. A JSONL manifest where each line is either:
       - a part record, or
       - a full manifest object.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

        if not records:
            raise ValueError(f"Empty JSONL file: {path}")

        # Case A: JSONL contains exactly one full manifest object.
        if len(records) == 1 and isinstance(records[0], dict) and "parts" in records[0]:
            return records[0]

        # Case B: JSONL contains one part per line.
        total_samples = 0
        for r in records:
            if "num_samples" not in r:
                raise ValueError(
                    f"JSONL part record in {path} does not contain 'num_samples'. "
                    f"Record keys: {list(r.keys())}"
                )
            total_samples += int(r["num_samples"])

        return {
            "prefix": path.stem,
            "total_samples": total_samples,
            "num_parts": len(records),
            "parts": records,
        }

    raise ValueError(f"Unsupported manifest extension: {path}")


def normalize_part_path(part: Dict[str, Any], manifest_path: Path, make_absolute: bool) -> Dict[str, Any]:
    """
    Normalize the 'path' field in each part record.
    If path is relative, make it relative to the manifest file's parent directory.
    """
    item = dict(part)

    if "path" not in item:
        raise KeyError(f"Part record in {manifest_path} does not contain 'path': {item.keys()}")

    p = Path(item["path"])

    if not p.is_absolute():
        p = manifest_path.parent / p

    if make_absolute:
        p = p.resolve()

    item["path"] = str(p)
    return item


def merge_manifests(
    root: Path,
    pattern: str,
    output: Path,
    prefix: str,
    make_absolute: bool,
    check_files: bool,
) -> Dict[str, Any]:
    manifest_paths = sorted(root.glob(pattern))

    if not manifest_paths:
        raise FileNotFoundError(f"No manifest files found with pattern: {root / pattern}")

    all_parts: List[Dict[str, Any]] = []
    total_samples = 0
    global_part = 0

    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path)

        source_prefix = manifest.get("prefix", manifest_path.parent.name)
        parts = manifest.get("parts", [])

        if not isinstance(parts, list):
            raise TypeError(f"'parts' must be a list in {manifest_path}")

        print(
            f"Reading {manifest_path} | "
            f"prefix={source_prefix} | "
            f"declared_samples={manifest.get('total_samples')} | "
            f"declared_parts={manifest.get('num_parts')} | "
            f"actual_parts={len(parts)}"
        )

        manifest_sample_count = 0

        for part in parts:
            item = normalize_part_path(part, manifest_path, make_absolute=make_absolute)

            if check_files and not Path(item["path"]).exists():
                raise FileNotFoundError(f"Missing .pt file: {item['path']}")

            num_samples = int(item["num_samples"])
            manifest_sample_count += num_samples

            item["source_manifest"] = str(manifest_path)
            item["source_prefix"] = source_prefix
            item["source_part"] = item.get("part")
            item["global_part"] = global_part

            all_parts.append(item)
            total_samples += num_samples
            global_part += 1

        declared_total = manifest.get("total_samples")
        if declared_total is not None and int(declared_total) != manifest_sample_count:
            print(
                f"WARNING: {manifest_path} declared total_samples={declared_total}, "
                f"but sum(parts.num_samples)={manifest_sample_count}"
            )

    merged = {
        "prefix": prefix,
        "total_samples": total_samples,
        "num_parts": len(all_parts),
        "num_source_manifests": len(manifest_paths),
        "source_manifests": [str(p) for p in manifest_paths],
        "parts": all_parts,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LAPA feature manifest files.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder that contains shard folders, e.g. /datasets/ssv2/nips/features",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="z_rgb_train_shard*_0/*_manifest.json",
        help="Glob pattern relative to --root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output merged manifest path.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="z_rgb_train_all",
        help="Prefix name stored in the merged manifest.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Keep paths relative if they were relative. Default is to convert paths to absolute.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check that every .pt path in the manifest exists.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser()
    output = Path(args.output).expanduser()

    merged = merge_manifests(
        root=root,
        pattern=args.pattern,
        output=output,
        prefix=args.prefix,
        make_absolute=not args.relative_paths,
        check_files=args.check_files,
    )

    print("\nSaved:", output)
    print("prefix:", merged["prefix"])
    print("num_source_manifests:", merged["num_source_manifests"])
    print("num_parts:", merged["num_parts"])
    print("total_samples:", merged["total_samples"])

    if merged["parts"]:
        print("first part path:", merged["parts"][0]["path"])
        print("last part path:", merged["parts"][-1]["path"])


if __name__ == "__main__":
    main()
