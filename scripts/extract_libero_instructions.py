import argparse
import json
import re
from pathlib import Path


def extract_instruction_from_episode_name(name: str) -> str:
    """
    Extract natural language instruction from LIBERO episode folder name.

    Examples:
    libero_10_KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_demo_0
    -> turn on the stove and put the moka pot on it

    libero_libero_90_LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo_demo_14
    -> pick up the tomato sauce and put it in the basket
    """

    original = name

    # Remove one or more leading libero_ prefixes:
    # libero_90_...
    # libero_libero_90_...
    while name.startswith("libero_"):
        name = name[len("libero_"):]

    # Remove benchmark prefix:
    # 10_, 90_, spatial_, object_, goal_
    name = re.sub(r"^(10|90|spatial|object|goal)_", "", name)

    # Remove scene prefix:
    # KITCHEN_SCENE3_
    # LIVING_ROOM_SCENE1_
    # STUDY_SCENE2_
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)

    # Remove trailing demo/demo_id patterns:
    # _demo_demo_0
    # _demo_0
    # _demo
    name = re.sub(r"_demo_demo_\d+$", "", name)
    name = re.sub(r"_demo_\d+$", "", name)
    name = re.sub(r"_demo$", "", name)

    instruction = name.replace("_", " ").strip()
    instruction = re.sub(r"\s+", " ", instruction)

    if not instruction:
        instruction = original

    return instruction


def is_libero_episode(episode_id: str) -> bool:
    return episode_id.lower().startswith("libero_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode-root",
        type=str,
        required=True,
        help="Folder containing LIBERO episode folders, e.g. /datasets/libero_ssv2/frames_train",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="train.json",
        help="Output SSV2-style JSON label file",
    )

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional output JSONL file with instruction field",
    )

    parser.add_argument(
        "--include-folder",
        action="store_true",
        help="Include folder path in JSONL records. Not used in SSV2-style JSON.",
    )

    args = parser.parse_args()

    episode_root = Path(args.episode_root)
    assert episode_root.exists(), f"Episode root not found: {episode_root}"

    ssv2_style_records = []
    jsonl_records = []

    for ep_dir in sorted(episode_root.iterdir()):
        if not ep_dir.is_dir():
            continue

        episode_id = ep_dir.name

        if not is_libero_episode(episode_id):
            continue

        instruction = extract_instruction_from_episode_name(episode_id)

        # This is the format similar to SSV2 train.json / validation.json
        ssv2_style_records.append({
            "id": episode_id,
            "label": instruction,
            "template": instruction,
            "placeholders": []
        })

        # Optional JSONL format, useful for debugging or later custom pipeline
        jsonl_record = {
            "id": episode_id,
            "video_id": episode_id,
            "instruction": instruction,
            "label": instruction,
            "template": instruction,
            "placeholders": []
        }

        if args.include_folder:
            jsonl_record["folder"] = str(ep_dir)

        jsonl_records.append(jsonl_record)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(ssv2_style_records, f, indent=2, ensure_ascii=False)

    print(f"Found {len(ssv2_style_records)} LIBERO episodes")
    print(f"Saved SSV2-style JSON: {output_json}")

    if args.output_jsonl is not None:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with output_jsonl.open("w", encoding="utf-8") as f:
            for r in jsonl_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved JSONL: {output_jsonl}")

    print("\nExamples:")
    for r in ssv2_style_records[:10]:
        print(r["id"], "=>", r["label"])


if __name__ == "__main__":
    main()