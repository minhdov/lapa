import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def safe_folder_name(name: str) -> str:
    """
    Make folder name safe and consistent.
    """
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def convert_libero_depth_to_ssv2(
    input_root: Path,
    output_root: Path,
    depth_leaf: str = "agentview_depth",
    copy_mode: str = "copy",
    skip_existing: bool = True,
):
    """
    Convert LIBERO depth folder structure to SSV2-like structure.

    Input:
        LIBERO_depth/
          libero_10/
            TASK_NAME/
              demo_0/
                agentview_depth/
                  000000.png
                  000001.png

    Output:
        libero_depth_ssv2/
          libero_10_TASK_NAME_demo_0/
            000000.png
            000001.png
    """

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    episode_dirs = []

    for main_task_dir in sorted(input_root.iterdir()):
        if not main_task_dir.is_dir():
            continue

        # Example: libero_10
        main_task_name = main_task_dir.name

        for subtask_dir in sorted(main_task_dir.iterdir()):
            if not subtask_dir.is_dir():
                continue

            # Example: KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo
            subtask_name = subtask_dir.name

            for demo_dir in sorted(subtask_dir.iterdir()):
                if not demo_dir.is_dir():
                    continue

                # Example: demo_0
                demo_name = demo_dir.name

                depth_dir = demo_dir / depth_leaf

                if depth_dir.exists() and depth_dir.is_dir():
                    episode_dirs.append(
                        {
                            "main_task": main_task_name,
                            "subtask": subtask_name,
                            "demo": demo_name,
                            "depth_dir": depth_dir,
                        }
                    )

    print(f"Found {len(episode_dirs)} episodes with depth images.")

    num_copied = 0
    num_skipped = 0
    num_empty = 0

    for ep in tqdm(episode_dirs, desc="Converting episodes"):
        main_task = ep["main_task"]
        subtask = ep["subtask"]
        demo = ep["demo"]
        depth_dir = ep["depth_dir"]

        output_folder_name = safe_folder_name(
            f"{main_task}_{subtask}_{demo}"
        )

        output_episode_dir = output_root / output_folder_name
        output_episode_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(depth_dir.glob("*.png"))

        if len(image_files) == 0:
            num_empty += 1
            continue

        for img_path in image_files:
            dst_path = output_episode_dir / img_path.name

            if skip_existing and dst_path.exists():
                num_skipped += 1
                continue

            if copy_mode == "copy":
                shutil.copy2(img_path, dst_path)
            elif copy_mode == "symlink":
                if dst_path.exists():
                    dst_path.unlink()
                dst_path.symlink_to(img_path.resolve())
            else:
                raise ValueError(f"Unknown copy_mode: {copy_mode}")

            num_copied += 1

    print("\nDone.")
    print(f"Episodes found : {len(episode_dirs)}")
    print(f"Images copied  : {num_copied}")
    print(f"Images skipped : {num_skipped}")
    print(f"Empty episodes : {num_empty}")
    print(f"Output root    : {output_root}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Path to LIBERO_depth root folder",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Path to output libero_depth_ssv2 folder",
    )

    parser.add_argument(
        "--depth-leaf",
        type=str,
        default="agentview_depth",
        help="Depth image folder name inside each demo",
    )

    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Use copy to duplicate files, or symlink to save disk space",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Overwrite existing files instead of skipping",
    )

    args = parser.parse_args()

    convert_libero_depth_to_ssv2(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        depth_leaf=args.depth_leaf,
        copy_mode=args.copy_mode,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()