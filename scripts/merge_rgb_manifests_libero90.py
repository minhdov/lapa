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
    parser.add_argument("--out_manifest", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="z_rgb_train_all")
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()

    all_parts = []
    source_manifests = []
    total_samples = 0
    global_part = 0

    for manifest_str in args.manifests:
        manifest_path = Path(manifest_str)
        manifest = load_manifest(manifest_path)

        source_manifests.append(str(manifest_path))

        parts = manifest.get("parts", [])
        print(f"Reading: {manifest_path}")
        print(f"  prefix: {manifest.get('prefix')}")
        print(f"  total_samples: {manifest.get('total_samples')}")
        print(f"  num_parts: {len(parts)}")

        for part in parts:
            p = dict(part)

            part_path = Path(p["path"])
            if not part_path.exists():
                raise FileNotFoundError(f"Feature part not found: {part_path}")

            p["source_manifest"] = str(manifest_path)
            p["source_prefix"] = manifest.get("prefix", "")
            p["source_part"] = p.get("part")
            p["global_part"] = global_part
            p["part"] = global_part

            all_parts.append(p)
            total_samples += int(p["num_samples"])
            global_part += 1

    out = {
        "prefix": args.prefix,
        "total_samples": total_samples,
        "num_parts": len(all_parts),
        "num_source_manifests": len(source_manifests),
        "source_manifests": source_manifests,
        "parts": all_parts,
    }

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("out_manifest:", out_path)
    print("total_samples:", total_samples)
    print("num_parts:", len(all_parts))


if __name__ == "__main__":
    main()