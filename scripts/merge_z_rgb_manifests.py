import json
from pathlib import Path


manifest_paths = [
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/features/z_rgb_train_shard0/z_rgb_train_shard0_model_1_4_rgb_manifest.json",
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/features/z_rgb_train_shard1/z_rgb_train_shard1_model_1_4_rgb_manifest.json",
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/features/z_rgb_train_shard2/z_rgb_train_shard2_model_1_4_rgb_manifest.json",
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/features/z_rgb_train_shard3/z_rgb_train_shard3_model_1_4_rgb_manifest.json",
]

output_path = Path(
    "/datasets/ssv2_libero90/model_1_4_rgb_ssv2/features/"
    "z_rgb_train_all_model_1_4_rgb_manifest.json"
)

all_parts = []
total_samples = 0

for manifest_path in manifest_paths:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    parts = manifest.get("parts", [])

    if len(parts) == 0:
        raise RuntimeError(f"No parts found in: {manifest_path}")

    print(f"Loading {manifest_path}")
    print(f"  parts: {len(parts)}")
    print(f"  total_samples: {manifest.get('total_samples')}")

    for part in parts:
        part = dict(part)

        part_path = Path(part["path"])
        if not part_path.exists():
            raise FileNotFoundError(f"Feature part not found: {part_path}")

        part["global_part"] = len(all_parts)
        all_parts.append(part)
        total_samples += int(part["num_samples"])

merged_manifest = {
    "model": "model_1_4_rgb",
    "prefix": "z_rgb_train_all_model_1_4_rgb",
    "total_samples": total_samples,
    "num_parts": len(all_parts),
    "feature_dir": str(output_path.parent),
    "source_manifests": manifest_paths,
    "parts": all_parts,
}

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(merged_manifest, f, indent=2)

print("=" * 80)
print(f"Saved merged manifest to: {output_path}")
print(f"num_parts: {len(all_parts)}")
print(f"total_samples: {total_samples}")
print("=" * 80)