import json
from pathlib import Path

inputs = [
    "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json",
    "/datasets/libero_ssv2/features/z_rgb_train_shard0_0/z_rgb_train_shard0_0_manifest.json",
    "/datasets/libero_ssv2/features/z_rgb_train_shard1_0/z_rgb_train_shard1_0_manifest.json",
    "/datasets/libero_ssv2/features/z_rgb_train_shard2_0/z_rgb_train_shard2_0_manifest.json",
    "/datasets/libero_ssv2/features/z_rgb_train_shard3_0/z_rgb_train_shard3_0_manifest.json",
]

output = Path("/datasets/ssv2_libero_stage25_model4/z_rgb_train_mixed_manifest.json")
output.parent.mkdir(parents=True, exist_ok=True)

merged = {
    "prefix": "z_rgb_train_mixed",
    "total_samples": 0,
    "num_parts": 0,
    "num_source_manifests": len(inputs),
    "source_manifests": inputs,
    "parts": [],
}

part_id = 0

for manifest_path in inputs:
    manifest_path = Path(manifest_path)
    print("Reading:", manifest_path)

    m = json.load(open(manifest_path, "r"))

    for part in m["parts"]:
        new_part = dict(part)
        new_part["part"] = part_id

        # Keep path absolute
        new_part["path"] = str(Path(new_part["path"]).resolve())

        # Optional, useful for debugging
        new_part["source_manifest"] = str(manifest_path)

        merged["parts"].append(new_part)
        merged["total_samples"] += int(new_part["num_samples"])
        part_id += 1

merged["num_parts"] = len(merged["parts"])

with output.open("w") as f:
    json.dump(merged, f, indent=2)

print("Saved:", output)
print("total_samples:", merged["total_samples"])
print("num_parts:", merged["num_parts"])