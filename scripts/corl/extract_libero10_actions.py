import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg"}


# ============================================================
# Basic parsing utils
# ============================================================

def normalize_name(s: str) -> str:
    """
    Normalize task/file names for robust matching.
    """
    s = Path(s).stem
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_video_id_from_path(path: str) -> str:
    """
    Example:
      /datasets/.../libero_10_KITCHEN_SCENE3_xxx_demo_demo_0/000000.png

    Returns:
      libero_10_KITCHEN_SCENE3_xxx_demo_demo_0
    """
    return Path(path).parent.name


def parse_frame_idx_from_path(path: str) -> int:
    """
    Example:
      000123.png -> 123
    """
    stem = Path(path).stem
    m = re.search(r"(\d+)$", stem)

    if not m:
        raise ValueError(f"Cannot parse frame index from path: {path}")

    return int(m.group(1))


def parse_libero_video_id(video_id: str) -> Tuple[str, int]:
    """
    Example:
      libero_10_KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_demo_0

    Returns:
      task_name = KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
      demo_idx = 0
    """
    m = re.search(r"_demo_demo_(\d+)$", video_id)

    if not m:
        raise ValueError(f"Cannot parse demo index from video_id: {video_id}")

    demo_idx = int(m.group(1))

    task_name = video_id[:m.start()]
    task_name = re.sub(r"^libero_10_", "", task_name)
    task_name = re.sub(r"^libero_90_", "", task_name)
    task_name = re.sub(r"^libero_", "", task_name)

    return task_name, demo_idx


# ============================================================
# HDF5 indexing
# ============================================================

def find_hdf5_files(hdf5_root: Path):
    files = []

    for ext in ["*.hdf5", "*.h5"]:
        files.extend(hdf5_root.rglob(ext))

    return sorted(files)


def build_task_to_hdf5_index(hdf5_root: Path) -> Dict[str, Path]:
    """
    Build mapping:
      normalized hdf5 filename -> hdf5 path

    Example:
      KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.hdf5

    key:
      kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it
    """
    hdf5_files = find_hdf5_files(hdf5_root)

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under: {hdf5_root}")

    index = {}

    for p in hdf5_files:
        key = normalize_name(p.stem)
        index[key] = p

    print(f"Found {len(hdf5_files)} hdf5 files under {hdf5_root}")

    return index


def resolve_hdf5_for_task(task_name: str, task_index: Dict[str, Path]) -> Optional[Path]:
    """
    Robust matching:
      1. exact normalized match
      2. task contains hdf5 stem
      3. hdf5 stem contains task
    """
    task_key = normalize_name(task_name)

    if task_key in task_index:
        return task_index[task_key]

    candidates = []

    for k, p in task_index.items():
        if task_key in k or k in task_key:
            candidates.append((k, p))

    if len(candidates) == 1:
        return candidates[0][1]

    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda x: len(x[0]), reverse=True)
        return candidates[0][1]

    return None


# ============================================================
# HDF5 action cache
# ============================================================

class HDF5ActionCache:
    def __init__(self, task_index: Dict[str, Path]):
        self.task_index = task_index
        self.cache = {}

    def get_actions(self, video_id: str):
        task_name, demo_idx = parse_libero_video_id(video_id)

        hdf5_path = resolve_hdf5_for_task(task_name, self.task_index)

        if hdf5_path is None:
            raise FileNotFoundError(
                f"Cannot find hdf5 for task_name={task_name}, video_id={video_id}"
            )

        cache_key = (str(hdf5_path), demo_idx)

        if cache_key in self.cache:
            return self.cache[cache_key]

        demo_key = f"demo_{demo_idx}"

        with h5py.File(hdf5_path, "r") as f:
            if "data" in f and demo_key in f["data"]:
                demo_group = f["data"][demo_key]
            elif demo_key in f:
                demo_group = f[demo_key]
            else:
                raise KeyError(
                    f"Cannot find {demo_key} in {hdf5_path}. "
                    f"Top-level keys: {list(f.keys())}"
                )

            if "actions" not in demo_group:
                raise KeyError(
                    f"No 'actions' key in {hdf5_path}/{demo_key}. "
                    f"Available keys: {list(demo_group.keys())}"
                )

            actions = np.asarray(demo_group["actions"], dtype=np.float32)

        self.cache[cache_key] = {
            "hdf5_path": str(hdf5_path),
            "demo_key": demo_key,
            "actions": actions,
        }

        return self.cache[cache_key]


# ============================================================
# Action magnitude
# ============================================================

def compute_xyz_magnitude(action: np.ndarray) -> float:
    """
    LIBERO action is usually [dx, dy, dz, droll, dpitch, dyaw, gripper].

    Here magnitude uses only translation:
      sqrt(dx^2 + dy^2 + dz^2)
    """
    xyz = action[:3]
    return float(np.linalg.norm(xyz))


# ============================================================
# Sample iterators
# ============================================================

def iter_samples_from_jsonl(input_jsonl: Path):
    """
    Input JSONL should have at least one of:
      image
      depth1_path
      depth_path
      rgb_path

    Optional:
      id
      video_id
    """
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue

            obj = json.loads(line)

            image_path = (
                obj.get("image")
                or obj.get("depth1_path")
                or obj.get("depth_path")
                or obj.get("rgb_path")
            )

            if image_path is None:
                raise KeyError(
                    f"Line {line_idx} has no image/depth1_path/depth_path/rgb_path field."
                )

            video_id = obj.get("video_id")
            if video_id is None:
                video_id = parse_video_id_from_path(image_path)

            sample_id = obj.get("id")
            if sample_id is None:
                frame_idx = parse_frame_idx_from_path(image_path)
                sample_id = f"{video_id}_{frame_idx:06d}"

            yield obj, sample_id, video_id, image_path


def iter_samples_from_image_root(image_root: Path):
    """
    image_root contains:
      image_root/video_id/000000.png
      image_root/video_id/000001.png
      ...
    """
    folders = sorted([p for p in image_root.iterdir() if p.is_dir()])

    for folder in folders:
        video_id = folder.name

        images = []
        for ext in IMG_EXTS:
            images.extend(folder.glob(f"*{ext}"))

        images = sorted(images)

        for img in images:
            frame_idx = parse_frame_idx_from_path(str(img))
            sample_id = f"{video_id}_{frame_idx:06d}"

            obj = {
                "id": sample_id,
                "video_id": video_id,
                "image": str(img),
            }

            yield obj, sample_id, video_id, str(img)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract LIBERO action_vector and translation magnitude "
            "from hdf5 and align with JSONL/image frames."
        )
    )

    parser.add_argument(
        "--hdf5_root",
        type=str,
        required=True,
        help="Root folder containing LIBERO-10 .hdf5/.h5 files.",
    )

    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="",
        help="Input JSONL with id/video_id/image or depth1_path.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Alternative to input_jsonl: root folder containing video folders with images.",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL with action_vector and magnitude added.",
    )

    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="If set, write samples with missing action as null instead of crashing.",
    )

    args = parser.parse_args()

    hdf5_root = Path(args.hdf5_root)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    task_index = build_task_to_hdf5_index(hdf5_root)
    action_cache = HDF5ActionCache(task_index)

    if args.input_jsonl:
        sample_iter = iter_samples_from_jsonl(Path(args.input_jsonl))
    elif args.image_root:
        sample_iter = iter_samples_from_image_root(Path(args.image_root))
    else:
        raise ValueError("You must provide either --input_jsonl or --image_root")

    total = 0
    missing = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for obj, sample_id, video_id, image_path in tqdm(sample_iter):
            total += 1

            frame_idx = parse_frame_idx_from_path(image_path)

            try:
                action_info = action_cache.get_actions(video_id)
                actions = action_info["actions"]

                if frame_idx >= len(actions):
                    raise IndexError(
                        f"frame_idx={frame_idx} >= len(actions)={len(actions)} "
                        f"for video_id={video_id}"
                    )

                action_vector = actions[frame_idx].astype(np.float32)
                magnitude = compute_xyz_magnitude(action_vector)

                obj["id"] = sample_id
                obj["video_id"] = video_id
                obj["image"] = image_path

                obj["action_vector"] = action_vector.tolist()
                obj["magnitude"] = magnitude
                obj["magnitude_mode"] = "xyz"
                obj["magnitude_definition"] = "sqrt(dx^2 + dy^2 + dz^2)"

                obj["action_frame_idx"] = frame_idx
                obj["action_hdf5_path"] = action_info["hdf5_path"]
                obj["action_demo_key"] = action_info["demo_key"]

            except Exception as e:
                if not args.allow_missing:
                    raise

                missing += 1

                obj["id"] = sample_id
                obj["video_id"] = video_id
                obj["image"] = image_path

                obj["action_vector"] = None
                obj["magnitude"] = None
                obj["magnitude_mode"] = "xyz"
                obj["magnitude_definition"] = "sqrt(dx^2 + dy^2 + dz^2)"
                obj["action_error"] = str(e)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done.")
    print("output_jsonl:", output_jsonl)
    print("total samples:", total)
    print("missing actions:", missing)
    print("loaded hdf5 demos:", len(action_cache.cache))


if __name__ == "__main__":
    main()