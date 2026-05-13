import json
import h5py
import torch
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


LIBERO_PREFIXES = [
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_spatial",
    "libero_object",
]


def parse_libero_image_path(image_path: str, hdf5_base_root: Path):
    """
    Example image path:
        libero_10_KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_demo_0/000000.png

    It will map to:
        hdf5_base_root / libero_10 / KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
    """
    p = Path(image_path)

    folder = p.parent.name
    frame_idx = int(p.stem)

    subset_name = None
    task_demo_name = None

    for prefix in LIBERO_PREFIXES:
        prefix_with_underscore = prefix + "_"
        if folder.startswith(prefix_with_underscore):
            subset_name = prefix
            task_demo_name = folder[len(prefix_with_underscore):]
            break

    if subset_name is None:
        raise ValueError(f"Cannot detect LIBERO subset from folder: {folder}")

    if "_demo_" not in task_demo_name:
        raise ValueError(f"Cannot parse demo name from image path: {image_path}")

    task_name, demo_suffix = task_demo_name.rsplit("_demo_", 1)
    demo_name = "demo_" + demo_suffix

    hdf5_path = hdf5_base_root / subset_name / f"{task_name}.hdf5"

    return hdf5_path, demo_name, frame_idx, subset_name


def build_id_to_action(jsonl_path: Path, hdf5_base_root: Path):
    id_to_action = {}
    hdf5_cache = {}

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Reading JSONL and HDF5 actions"):
            item = json.loads(line)

            sample_id = item["id"]
            image_path = item["image"]

            hdf5_path, demo_name, frame_idx, subset_name = parse_libero_image_path(
                image_path=image_path,
                hdf5_base_root=hdf5_base_root,
            )

            if hdf5_path not in hdf5_cache:
                if not hdf5_path.exists():
                    raise FileNotFoundError(
                        f"HDF5 file not found: {hdf5_path}\n"
                        f"subset={subset_name}, image={image_path}"
                    )

                hdf5_cache[hdf5_path] = h5py.File(hdf5_path, "r")

            h5 = hdf5_cache[hdf5_path]

            if demo_name not in h5["data"]:
                raise KeyError(f"{demo_name} not found in {hdf5_path}")

            actions = h5["data"][demo_name]["actions"]

            if frame_idx >= actions.shape[0]:
                raise IndexError(
                    f"frame_idx={frame_idx} >= actions length={actions.shape[0]} "
                    f"for {hdf5_path}, {demo_name}"
                )

            action_t = actions[frame_idx].astype("float32")
            id_to_action[sample_id] = action_t

    for h5 in hdf5_cache.values():
        h5.close()

    return id_to_action


def update_pt_files(feature_dir: Path, out_feature_dir: Path, id_to_action: dict):
    out_feature_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(feature_dir.glob("*.pt"))

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in: {feature_dir}")

    for pt_path in tqdm(pt_files, desc="Updating .pt files"):
        obj = torch.load(pt_path, map_location="cpu")

        if "id" not in obj:
            raise KeyError(f"'id' key not found in {pt_path}")

        ids = obj["id"]

        actions = []
        missing_ids = []

        for sample_id in ids:
            if sample_id not in id_to_action:
                missing_ids.append(sample_id)
            else:
                actions.append(id_to_action[sample_id])

        if missing_ids:
            print("Example missing ids:", missing_ids[:10])
            raise KeyError(
                f"{len(missing_ids)} ids from {pt_path.name} "
                f"not found in JSONL/HDF5 mapping"
            )

        obj["action_t"] = torch.tensor(actions, dtype=torch.float32)

        out_path = out_feature_dir / pt_path.name
        torch.save(obj, out_path)

        print(
            "saved:",
            out_path,
            "| action_t:",
            tuple(obj["action_t"].shape),
            "| z_depth_feature:",
            tuple(obj["z_depth_feature"].shape) if "z_depth_feature" in obj else "not found",
        )

    for manifest_path in feature_dir.glob("*manifest*.json"):
        shutil.copy2(manifest_path, out_feature_dir / manifest_path.name)
        print("copied manifest:", manifest_path.name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--out_feature_dir", type=str, required=True)

    parser.add_argument(
        "--hdf5_base_root",
        type=str,
        required=True,
        help="Base folder containing libero_10, libero_90, libero_goal, libero_spatial, libero_object",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    feature_dir = Path(args.feature_dir)
    out_feature_dir = Path(args.out_feature_dir)
    hdf5_base_root = Path(args.hdf5_base_root)

    print("jsonl_path:", jsonl_path)
    print("feature_dir:", feature_dir)
    print("out_feature_dir:", out_feature_dir)
    print("hdf5_base_root:", hdf5_base_root)

    id_to_action = build_id_to_action(
        jsonl_path=jsonl_path,
        hdf5_base_root=hdf5_base_root,
    )

    print("Loaded action_t for samples:", len(id_to_action))

    update_pt_files(
        feature_dir=feature_dir,
        out_feature_dir=out_feature_dir,
        id_to_action=id_to_action,
    )

    print("Done.")


if __name__ == "__main__":
    main()