#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def find_depth_keys(h5_file):
    depth_keys = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and "depth" in name.lower():
            depth_keys.append(name)

    h5_file.visititems(visitor)
    return depth_keys


def print_hdf5_structure(hdf5_path):
    hdf5_path = Path(hdf5_path)
    print(f"\n[INSPECT] {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name:90s} shape={obj.shape} dtype={obj.dtype}")

        f.visititems(visitor)


def squeeze_depth_frame(depth):
    depth = np.asarray(depth)

    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError(f"Unsupported depth frame shape: {depth.shape}")

    if depth.ndim != 2:
        raise ValueError(f"Expected depth frame [H, W], got {depth.shape}")

    return depth.astype(np.float32)


def normalize_to_uint16_like_depth_anything(depth):
    """
    Match your Depth Anything V2 saving code:

        depth = (depth - d_min) / (d_max - d_min)
        depth_uint16 = depth * 65535

    Input should already have the correct direction:
        near = larger value
        far  = smaller value
    """
    depth = depth.astype(np.float32)

    valid = np.isfinite(depth)

    if valid.sum() < 10:
        return np.zeros(depth.shape, dtype=np.uint16)

    d_min = float(depth[valid].min())
    d_max = float(depth[valid].max())

    if d_max - d_min < 1e-8:
        return np.zeros(depth.shape, dtype=np.uint16)

    out = np.zeros_like(depth, dtype=np.float32)
    out[valid] = (depth[valid] - d_min) / (d_max - d_min)
    out = np.clip(out, 0.0, 1.0)

    return (out * 65535.0).astype(np.uint16)


def libero_metric_depth_to_da_style_uint16(depth, eps=1e-6):
    """
    Convert LIBERO metric depth to Depth Anything-like saved PNG.

    LIBERO metric depth:
        near = small value
        far  = large value

    Depth Anything saved format in your script:
        larger predicted depth value becomes brighter after min-max normalize.

    So we first inverse:
        inv_depth = 1 / depth

    Then apply the exact same min-max-to-uint16 style.
    """
    depth = squeeze_depth_frame(depth)

    valid = np.isfinite(depth) & (depth > 0)

    inv_depth = np.zeros_like(depth, dtype=np.float32)
    inv_depth[valid] = 1.0 / (depth[valid] + eps)

    return normalize_to_uint16_like_depth_anything(inv_depth)


def iter_depth_frames(depth_array):
    depth_array = np.asarray(depth_array)

    if depth_array.ndim == 2:
        yield 0, depth_array
        return

    if depth_array.ndim == 3:
        # [T, H, W] or [H, W, 1]
        if depth_array.shape[-1] == 1:
            yield 0, depth_array
        else:
            for i in range(depth_array.shape[0]):
                yield i, depth_array[i]
        return

    if depth_array.ndim == 4:
        # [T, H, W, 1] or [T, 1, H, W]
        for i in range(depth_array.shape[0]):
            yield i, depth_array[i]
        return

    raise ValueError(f"Unsupported depth array shape: {depth_array.shape}")


def safe_name(name):
    return name.replace("/", "__").replace(" ", "_")


def extract_one_hdf5(
    hdf5_path,
    output_root,
    preferred_key=None,
    save_all_depth_keys=False,
):
    hdf5_path = Path(hdf5_path)
    output_root = Path(output_root)

    task_name = hdf5_path.stem

    with h5py.File(hdf5_path, "r") as f:
        depth_keys = find_depth_keys(f)

        if len(depth_keys) == 0:
            print(f"[WARN] No depth key found: {hdf5_path}")
            return

        if preferred_key is not None:
            selected_keys = [k for k in depth_keys if preferred_key in k]
        else:
            selected_keys = depth_keys

        if len(selected_keys) == 0:
            print(f"[WARN] preferred_key='{preferred_key}' not found in {hdf5_path.name}")
            print("Available depth keys:")
            for k in depth_keys:
                print(f"  {k} shape={f[k].shape} dtype={f[k].dtype}")
            return

        if not save_all_depth_keys:
            selected_keys = selected_keys[:1]

        print(f"\n[FILE] {hdf5_path.name}")
        for key in selected_keys:
            print(f"  using key: {key} shape={f[key].shape} dtype={f[key].dtype}")

            depth_array = f[key][()]
            out_dir = output_root / task_name / safe_name(key)
            out_dir.mkdir(parents=True, exist_ok=True)

            total = depth_array.shape[0] if np.asarray(depth_array).ndim >= 3 else 1

            for frame_idx, depth_frame in tqdm(
                iter_depth_frames(depth_array),
                total=total,
                desc=f"Saving {task_name}",
                leave=False,
            ):
                depth_png = libero_metric_depth_to_da_style_uint16(depth_frame)

                out_path = out_dir / f"{frame_idx:06d}.png"

                ok = cv2.imwrite(str(out_path), depth_png)
                if not ok:
                    print(f"[WARN] Failed to save: {out_path}")

            print(f"[OK] Saved to: {out_dir}")


def extract_folder(
    input_dir,
    output_dir,
    preferred_key=None,
    save_all_depth_keys=False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    hdf5_files = sorted(input_dir.rglob("*.hdf5"))

    print(f"[INFO] input_dir : {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] hdf5 files: {len(hdf5_files)}")

    for hdf5_path in tqdm(hdf5_files, desc="Processing hdf5"):
        extract_one_hdf5(
            hdf5_path=hdf5_path,
            output_root=output_dir,
            preferred_key=preferred_key,
            save_all_depth_keys=save_all_depth_keys,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--preferred_key",
        type=str,
        default=None,
        help="Example: agentview_depth or robot0_eye_in_hand_depth",
    )

    parser.add_argument(
        "--save_all_depth_keys",
        action="store_true",
    )

    parser.add_argument(
        "--inspect_one",
        type=str,
        default=None,
        help="Path to one hdf5 file for checking keys.",
    )

    args = parser.parse_args()

    if args.inspect_one is not None:
        print_hdf5_structure(args.inspect_one)
        return

    extract_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preferred_key=args.preferred_key,
        save_all_depth_keys=args.save_all_depth_keys,
    )


if __name__ == "__main__":
    main()