import argparse
import json
from pathlib import Path


def load_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssv2_manifest", type=str, required=True)
    parser.add_argument("--libero90_manifest", type=str, required=True)
    parser.add_argument("--out_manifest", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="z_rgb_train_mixed")
    args = parser.parse_args()

    source_manifest_paths = [
        args.ssv2_manifest,
        args.libero90_manifest,
    ]

    all_parts = []
    total_samples = 0
    global_part = 0

    for source_name, manifest_path_str in [
        ("ssv2", args.ssv2_manifest),
        ("libero90", args.libero90_manifest),
    ]:
        manifest_path = Path(manifest_path_str)
        manifest = load_manifest(manifest_path)

        print(f"Reading {source_name}: {manifest_path}")
        print(f"  prefix: {manifest.get('prefix')}")
        print(f"  total_samples: {manifest.get('total_samples')}")
        print(f"  num_parts: {manifest.get('num_parts')}")
        print(f"  actual parts: {len(manifest.get('parts', []))}")

        for part in manifest["parts"]:
            p = dict(part)

            if "path" not in p:
                raise KeyError(f"Missing 'path' in part from {manifest_path}")

            part_path = Path(p["path"])
            if not part_path.exists():
                raise FileNotFoundError(f"Feature .pt not found: {part_path}")

            p["source_dataset"] = source_name
            p["source_manifest"] = str(manifest_path)
            p["source_prefix"] = manifest.get("prefix", "")
            p["source_part"] = p.get("part")
            p["part"] = global_part
            p["global_part"] = global_part

            all_parts.append(p)
            total_samples += int(p["num_samples"])
            global_part += 1

    out = {
        "prefix": args.prefix,
        "total_samples": total_samples,
        "num_parts": len(all_parts),
        "num_source_manifests": len(source_manifest_paths),
        "source_manifests": source_manifest_paths,
        "parts": all_parts,
    }

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("out_manifest:", out_manifest)
    print("total_samples:", total_samples)
    print("num_parts:", len(all_parts))


if __name__ == "__main__":
    main()