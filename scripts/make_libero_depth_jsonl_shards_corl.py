from pathlib import Path
import argparse
import json


def sort_key_path(p: Path):
    stem = p.stem
    return int(stem) if stem.isdigit() else stem


def get_image_files(folder: Path):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=sort_key_path,
    )


def make_depth_jsonl_shards(
    depth_root: Path,
    out_dir: Path,
    num_shards: int,
    prefix: str,
):
    if not depth_root.exists():
        raise FileNotFoundError(f"Depth root not found: {depth_root}")

    folders = sorted([p for p in depth_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    if len(folders) == 0:
        raise RuntimeError(f"No episode folders found in: {depth_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(folders)
    total_lines = 0

    for shard_id in range(num_shards):
        start = shard_id * n // num_shards
        end = (shard_id + 1) * n // num_shards
        shard_folders = folders[start:end]

        out_path = out_dir / f"{prefix}_shard{shard_id}.jsonl"

        num_lines = 0
        num_skipped = 0

        with out_path.open("w", encoding="utf-8") as f:
            for ep_dir in shard_folders:
                video_id = ep_dir.name
                image_files = get_image_files(ep_dir)

                if len(image_files) == 0:
                    print(f"[Skip] no images: {ep_dir}")
                    num_skipped += 1
                    continue

                for frame_idx, img_path in enumerate(image_files):
                    record = {
                        "id": f"{video_id}_{img_path.stem}",
                        "video_id": video_id,
                        "image": str(img_path),
                        "frame_index": frame_idx,
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_lines += 1

        total_lines += num_lines

        print(
            f"shard {shard_id}: "
            f"{len(shard_folders)} folders | "
            f"{num_lines} lines | "
            f"skipped={num_skipped} | "
            f"{out_path}"
        )

    print(f"Done. Total folders: {n}")
    print(f"Done. Total lines: {total_lines}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--depth_root",
        type=str,
        default="/datasets/libero_ssv2/libero_depth_ssv2",
        help="Folder containing LIBERO depth episode folders.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/datasets/libero_ssv2",
        help="Output directory for depth_train_shard*.jsonl files.",
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Number of shards.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="depth_train",
        help="Output prefix. Default creates depth_train_shard0.jsonl.",
    )

    args = parser.parse_args()

    make_depth_jsonl_shards(
        depth_root=Path(args.depth_root),
        out_dir=Path(args.out_dir),
        num_shards=args.num_shards,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()