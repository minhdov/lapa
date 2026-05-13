import re
import torch
import h5py
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
BASE = Path("/datasets/libero2")
SHARD = "z_depth_train_shard0"

stage1_dir = BASE / "features_depth_stage1" / SHARD
model2_dir = BASE / "features_stage25_model2_both_libero_ssv2" / SHARD
model4_dir = BASE / "features_stage25_model4_both_libero_ssv2" / SHARD

hdf5_root = Path("/datasets/LIBERO/datasets")

output_dir = BASE / "features_merged_stage1_model2_model4_hdf5_actions_libero" / SHARD
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def check_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def find_feature_key(obj, file_name="unknown"):
    candidate_keys = [
        "feature_output",
        "z_rgb_feature",
        "z_depth_feature",
        "features",
        "feature",
        "hidden_states",
    ]

    for k in candidate_keys:
        if k in obj and torch.is_tensor(obj[k]) and obj[k].ndim == 2:
            return k

    tensor_keys = [
        k for k, v in obj.items()
        if torch.is_tensor(v) and v.ndim == 2
    ]

    if len(tensor_keys) == 1:
        return tensor_keys[0]

    raise KeyError(
        f"Cannot identify feature key for {file_name}.\n"
        f"Available keys: {list(obj.keys())}\n"
        f"2D tensor keys: {tensor_keys}"
    )


def parse_demo_id(text):
    if text is None:
        return None

    m = re.search(r"demo[_\-]?(\d+)", str(text))
    if m:
        return int(m.group(1))

    return None


def parse_frame_id(text):
    if text is None:
        return None

    text = str(text)

    patterns = [
        r"img[_\-]?(\d+)",
        r"frame[_\-]?(\d+)",
        r"image[_\-]?(\d+)",
        r"rgb[_\-]?(\d+)",
        r"depth[_\-]?(\d+)",
        r"/(\d{1,6})\.(png|jpg|jpeg)$",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))

    return None


def find_hdf5_files(root: Path):
    files = []
    for ext in ["*.hdf5", "*.h5"]:
        files.extend(root.rglob(ext))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {root}")

    return files


def build_demo_to_hdf5_index(hdf5_files):
    demo_to_files = {}

    for h5_path in hdf5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                if "data" not in f:
                    continue

                for demo_key in f["data"].keys():
                    demo_id = parse_demo_id(demo_key)
                    if demo_id is None:
                        continue

                    action_path = f"data/{demo_key}/actions"
                    if action_path in f:
                        if demo_id not in demo_to_files:
                            demo_to_files[demo_id] = []
                        demo_to_files[demo_id].append(h5_path)

        except Exception as e:
            print(f"WARNING: cannot read {h5_path}: {e}")

    return demo_to_files


def choose_hdf5_for_sample(sample_text, candidate_files):
    sample_text = str(sample_text)

    if len(candidate_files) == 1:
        return candidate_files[0]

    for p in candidate_files:
        if p.stem in sample_text:
            return p

    best_file = None
    best_score = -1

    for p in candidate_files:
        tokens = [t for t in p.stem.split("_") if len(t) >= 4]
        score = sum(1 for t in tokens if t in sample_text)

        if score > best_score:
            best_score = score
            best_file = p

    if best_file is not None and best_score >= 3:
        return best_file

    raise RuntimeError(
        "Multiple HDF5 candidates found, but cannot choose the correct one.\n"
        f"Sample text: {sample_text}\n"
        f"Candidates: {[str(p) for p in candidate_files[:20]]}"
    )


def get_action_from_hdf5(sample_id, image_path, depth_pair, video_id, demo_to_files):
    candidate_texts = [sample_id, image_path, depth_pair, video_id]

    joined_text = " ".join(
        [str(x) for x in candidate_texts if x is not None]
    )

    demo_id = None
    frame_id = None

    for x in candidate_texts:
        if x is None:
            continue

        if demo_id is None:
            demo_id = parse_demo_id(x)

        if frame_id is None:
            frame_id = parse_frame_id(x)

    if demo_id is None:
        raise ValueError(
            "Cannot parse demo id from sample.\n"
            f"id={sample_id}\n"
            f"video_id={video_id}\n"
            f"image={image_path}\n"
            f"depth_pair={depth_pair}"
        )

    if frame_id is None:
        raise ValueError(
            "Cannot parse frame/timestep from sample.\n"
            f"id={sample_id}\n"
            f"video_id={video_id}\n"
            f"image={image_path}\n"
            f"depth_pair={depth_pair}"
        )

    if demo_id not in demo_to_files:
        raise KeyError(f"demo_{demo_id} not found in any HDF5 file")

    h5_path = choose_hdf5_for_sample(joined_text, demo_to_files[demo_id])

    with h5py.File(h5_path, "r") as f:
        possible_demo_keys = [
            k for k in f["data"].keys()
            if parse_demo_id(k) == demo_id
        ]

        if not possible_demo_keys:
            raise KeyError(f"Cannot find demo_{demo_id} in {h5_path}")

        demo_key = possible_demo_keys[0]
        actions = f[f"data/{demo_key}/actions"]

        if frame_id >= actions.shape[0]:
            raise IndexError(
                f"frame_id {frame_id} out of range.\n"
                f"hdf5: {h5_path}\n"
                f"demo_key: {demo_key}\n"
                f"actions shape: {actions.shape}"
            )

        action = actions[frame_id]

    return torch.tensor(action, dtype=torch.float32)


def get_list_value(obj, key, i):
    if key not in obj:
        return None
    if obj[key] is None:
        return None
    return obj[key][i]


def get_part_id_from_stage1_file(path: Path):
    m = re.search(r"part(\d+)\.pt$", path.name)
    if not m:
        raise ValueError(f"Cannot parse part id from {path.name}")
    return int(m.group(1))


def process_one_part(stage1_file: Path, demo_to_files):
    part_id = get_part_id_from_stage1_file(stage1_file)
    part_name = f"part{part_id:05d}"

    model2_file = model2_dir / f"{SHARD}_model2_{part_name}.pt"
    model4_file = model4_dir / f"{SHARD}_model4_{part_name}.pt"
    output_file = output_dir / f"{SHARD}_merged_{part_name}.pt"

    check_exists(stage1_file)
    check_exists(model2_file)
    check_exists(model4_file)

    print("\n" + "=" * 100)
    print(f"Processing {part_name}")
    print("Stage1 file:", stage1_file)
    print("Model2 file:", model2_file)
    print("Model4 file:", model4_file)
    print("Output file:", output_file)

    print("\nLoading .pt files...")
    stage1 = torch.load(stage1_file, map_location="cpu")
    model2 = torch.load(model2_file, map_location="cpu")
    model4 = torch.load(model4_file, map_location="cpu")

    if "z_depth_indices" not in stage1:
        raise KeyError("stage1 file does not contain 'z_depth_indices'")

    if "z_depth_feature" not in stage1:
        raise KeyError("stage1 file does not contain 'z_depth_feature'")

    model2_feature_key = find_feature_key(model2, "model2")
    model4_feature_key = find_feature_key(model4, "model4")

    n = min(len(stage1["id"]), len(model2["id"]), len(model4["id"]))

    print("Number of samples:")
    print("  stage1:", len(stage1["id"]))
    print("  model2:", len(model2["id"]))
    print("  model4:", len(model4["id"]))
    print("  using:", n)

    if stage1["id"][:n] != model2["id"][:n]:
        print("First 5 stage1 ids:", stage1["id"][:5])
        print("First 5 model2 ids:", model2["id"][:5])
        raise RuntimeError(f"{part_name}: stage1 and model2 IDs are not aligned by order")

    if stage1["id"][:n] != model4["id"][:n]:
        print("First 5 stage1 ids:", stage1["id"][:5])
        print("First 5 model4 ids:", model4["id"][:5])
        raise RuntimeError(f"{part_name}: stage1 and model4 IDs are not aligned by order")

    print("Reading actions directly from HDF5...")
    actions_from_hdf5 = []

    for i in range(n):
        sample_id = stage1["id"][i]
        video_id = get_list_value(stage1, "video_id", i)
        image_path = get_list_value(stage1, "image", i)
        depth_pair = get_list_value(stage1, "depth_pair", i)

        action = get_action_from_hdf5(
            sample_id=sample_id,
            image_path=image_path,
            depth_pair=depth_pair,
            video_id=video_id,
            demo_to_files=demo_to_files,
        )

        actions_from_hdf5.append(action)

        if i < 2:
            print("\nSample", i)
            print("  id:", sample_id)
            print("  video_id:", video_id)
            print("  action from hdf5:", action)

        if (i + 1) % 1000 == 0:
            print(f"  Read {i + 1}/{n} actions")

    actions_from_hdf5 = torch.stack(actions_from_hdf5, dim=0)

    # action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
    # magnitude uses translation only: dx, dy, dz
    magnitude = torch.linalg.norm(actions_from_hdf5[:, :3].float(), dim=1)

    merged = {
        "id": stage1["id"][:n],
        "video_id": stage1.get("video_id", None)[:n]
        if stage1.get("video_id", None) is not None
        else None,
        "image": stage1.get("image", None)[:n]
        if stage1.get("image", None) is not None
        else None,
        "depth_pair": stage1.get("depth_pair", None)[:n]
        if stage1.get("depth_pair", None) is not None
        else None,

        # Stage 1 outputs
        "z_depth_indices": stage1["z_depth_indices"][:n],
        "z_depth_feature": stage1["z_depth_feature"][:n],

        # Model 2 and Model 4 features
        "feature_model2": model2[model2_feature_key][:n],
        "feature_model4": model4[model4_feature_key][:n],

        # HDF5 action and xyz-only magnitude
        "action_vector": actions_from_hdf5,
        "magnitude": magnitude,
    }

    print("Saving merged file...")
    torch.save(merged, output_file)

    print("Saved:", output_file)
    print("Merged content:")
    for k, v in merged.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
        else:
            print(f"  {k}: {type(v)}")

    return output_file, n


def main():
    print("Stage1 dir:", stage1_dir)
    print("Model2 dir:", model2_dir)
    print("Model4 dir:", model4_dir)
    print("HDF5 root:", hdf5_root)
    print("Output dir:", output_dir)

    print("\nSearching HDF5 files...")
    hdf5_files = find_hdf5_files(hdf5_root)
    print("num hdf5 files:", len(hdf5_files))
    print("first hdf5 files:")
    for p in hdf5_files[:5]:
        print(" ", p)

    print("\nBuilding demo -> HDF5 index...")
    demo_to_files = build_demo_to_hdf5_index(hdf5_files)
    print("num demo ids indexed:", len(demo_to_files))

    stage1_files = sorted(stage1_dir.glob(f"{SHARD}_stage1_part*.pt"))

    if not stage1_files:
        raise FileNotFoundError(f"No stage1 part files found in {stage1_dir}")

    print("\nFound stage1 parts:", len(stage1_files))
    for p in stage1_files[:10]:
        print(" ", p.name)
    if len(stage1_files) > 10:
        print(" ...")

    total_samples = 0
    output_files = []

    for stage1_file in stage1_files:
        out_file, n = process_one_part(stage1_file, demo_to_files)
        output_files.append(out_file)
        total_samples += n

    print("\n" + "=" * 100)
    print("DONE")
    print("Generated files:", len(output_files))
    print("Total samples:", total_samples)
    print("Output dir:", output_dir)
    for p in output_files[:10]:
        print(" ", p.name)
    if len(output_files) > 10:
        print(" ...")


if __name__ == "__main__":
    main()