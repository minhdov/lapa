#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def squeeze_depth_frame(depth):
    """
    Convert depth frame to [H, W] float32.

    Supported shapes:
    - [H, W]
    - [H, W, 1]
    - [1, H, W]
    """
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
    Normalize one inverse-depth frame to uint16 PNG.

    Output:
    - uint16 image in range [0, 65535]
    - near objects should have larger/brighter values
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
    Convert LIBERO metric depth to Depth-Anything-like uint16 PNG.

    LIBERO metric depth:
        near = small value
        far  = large value

    Depth Anything saved depth style:
        near = larger/brighter value
        far  = smaller/darker value

    Therefore:
        inverse depth = 1 / metric_depth
    """
    depth = squeeze_depth_frame(depth)

    valid = np.isfinite(depth) & (depth > 0)

    inv_depth = np.zeros_like(depth, dtype=np.float32)
    inv_depth[valid] = 1.0 / (depth[valid] + eps)

    return normalize_to_uint16_like_depth_anything(inv_depth)


def save_depth_png(depth, save_path: Path, overwrite=False):
    """
    Save one depth frame as uint16 PNG.
    """
    if save_path.exists() and not overwrite:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    depth_png = libero_metric_depth_to_da_style_uint16(depth)

    ok = cv2.imwrite(str(save_path), depth_png)
    if not ok:
        raise RuntimeError(f"Failed to save depth PNG: {save_path}")


def extract_one_hdf5(
    hdf5_path: Path,
    dataset_root: Path,
    output_root: Path,
    camera_keys,
    overwrite=False,
):
    """
    Extract depth images from one LIBERO hdf5 file.

    Output structure is similar to RGB extraction:

    output_root / task_group / task_name / demo_name / camera_key / 000000.png
    """
    relative_path = hdf5_path.relative_to(dataset_root)
    task_group = relative_path.parent
    task_name = hdf5_path.stem

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"[SKIP] No data group: {hdf5_path}")
            return

        demos = sorted(
            f["data"].keys(),
            key=lambda x: int(x.replace("demo_", "")) if x.startswith("demo_") else x
        )

        for demo_name in demos:
            if "obs" not in f["data"][demo_name]:
                print(f"[SKIP] No obs group: {hdf5_path} / {demo_name}")
                continue

            obs_group = f["data"][demo_name]["obs"]

            for camera_key in camera_keys:
                if camera_key not in obs_group:
                    continue

                depth_array = obs_group[camera_key]

                save_dir = output_root / task_group / task_name / demo_name / camera_key
                save_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"[EXTRACT] {hdf5_path.name} | {demo_name} | {camera_key} | "
                    f"shape={depth_array.shape}, dtype={depth_array.dtype}"
                )

                for i in range(depth_array.shape[0]):
                    save_path = save_dir / f"{i:06d}.png"

                    if save_path.exists() and not overwrite:
                        continue

                    save_depth_png(
                        depth=depth_array[i],
                        save_path=save_path,
                        overwrite=overwrite,
                    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to LIBERO depth hdf5 root folder",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to output depth PNG folder",
    )

    parser.add_argument(
        "--camera_keys",
        type=str,
        nargs="+",
        default=["agentview_depth", "eye_in_hand_depth"],
        help="Depth keys to extract",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    hdf5_files = sorted(dataset_root.rglob("*.hdf5"))

    print(f"Dataset root: {dataset_root}")
    print(f"Output root : {output_root}")
    print(f"Found {len(hdf5_files)} hdf5 files")
    print(f"Camera keys : {args.camera_keys}")

    for hdf5_path in tqdm(hdf5_files, desc="Extracting HDF5 files"):
        extract_one_hdf5(
            hdf5_path=hdf5_path,
            dataset_root=dataset_root,
            output_root=output_root,
            camera_keys=args.camera_keys,
            overwrite=args.overwrite,
        )

    print("Done.")


if __name__ == "__main__":
    main()