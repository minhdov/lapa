from pathlib import Path
import argparse


def sort_key(name: str):
    return int(name) if name.isdigit() else name


def split_folders(root: Path, out_dir: Path, num_shards: int, prefix: str):
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    folders = sorted(
        [p.name for p in root.iterdir() if p.is_dir()],
        key=sort_key,
    )

    if len(folders) == 0:
        raise RuntimeError(f"No folders found in: {root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(folders)

    for i in range(num_shards):
        start = i * n // num_shards
        end = (i + 1) * n // num_shards
        shard = folders[start:end]

        out_path = out_dir / f"{prefix}_shard{i}.txt"
        out_path.write_text("\n".join(shard) + "\n", encoding="utf-8")

        print(f"shard {i}: {len(shard)} folders | {shard[0]} -> {shard[-1]} | {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Split video folders into shard txt files.")
    parser.add_argument(
        "--root",
        type=str,
        default="/storage/minh/philo/datasets/libero_ssv2/libero_images_ssv2",
        help="Folder containing video folders, e.g. frames_train.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/storage/minh/philo/datasets/libero_ssv2/shards",
        help="Output folder for shard txt files.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=4,
        help="Number of shards to create.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="train_folders",
        help="Output filename prefix.",
    )

    args = parser.parse_args()

    split_folders(
        root=Path(args.root),
        out_dir=Path(args.out_dir),
        num_shards=args.num_shards,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
