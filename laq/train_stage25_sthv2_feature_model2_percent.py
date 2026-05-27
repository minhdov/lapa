import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision.utils import save_image

from laq_model.latent_action_quantization_stage25_feature import (
    LatentActionQuantizationStage25,
)

from laq_model.laq_stage25_trainer_feature import (
    LAQStage25Trainer,
)

from laq_model.data_stage25_feature import (
    Stage25Dataset,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_percent",
        type=float,
        required=True,
        help="Percent of dataset to use, e.g. 5, 10, 20, 80",
    )

    parser.add_argument(
        "--z_rgb_feature_manifest",
        type=str,
        default="/datasets/ssv2_libero90/stage1_model1_1_depth_ssv2/features/z_rgb_train_all_manifest.json",
    )

    parser.add_argument(
        "--z_depth_path",
        type=str,
        default="/datasets/ssv2_libero90/stage1_model1_1_depth_ssv2/z_depth_train.jsonl",
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="results_stage25_final_feature_model2_ssv2",
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=65001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_model_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument(
        "--subset_mode",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Use first N samples or random N samples",
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def make_subset(dataset, data_percent, mode="first", seed=42):
    assert 0 < data_percent <= 100, "data_percent must be in (0, 100]"

    total_len = len(dataset)
    subset_len = int(total_len * data_percent / 100.0)

    # avoid zero samples for very small datasets
    subset_len = max(1, subset_len)

    if mode == "first":
        indices = list(range(subset_len))

    elif mode == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(total_len, generator=g)[:subset_len].tolist()

    else:
        raise ValueError(f"Unknown subset mode: {mode}")

    return Subset(dataset, indices), total_len, subset_len


def main():
    args = parse_args()

    percent_str = str(args.data_percent).replace(".", "p")
    results_folder = f"{args.results_root}_{percent_str}percent"

    dataset_full = Stage25Dataset(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        image_size=256,
        code_seq_len=4,
        repeat_depth_to_3ch=True,
        depth_scale=65535.0,
        check_length_alignment=True,
        keep_z_rgb_indices=False,
    )

    dataset, total_len, subset_len = make_subset(
        dataset_full,
        data_percent=args.data_percent,
        mode=args.subset_mode,
        seed=args.seed,
    )

    print("=" * 80)
    print("Stage 2.5 training with dataset percent")
    print("z_rgb_feature_manifest:", args.z_rgb_feature_manifest)
    print("z_depth_path:", args.z_depth_path)
    print("data_percent:", args.data_percent)
    print("subset_mode:", args.subset_mode)
    print("total dataset size:", total_len)
    print("used dataset size:", subset_len)
    print("results_folder:", results_folder)
    print("=" * 80)

    # Debug one sample
    sample = dataset[0]
    print("id:", sample["id"])
    print(
        "depth1:",
        sample["depth1"].shape,
        sample["depth1"].min(),
        sample["depth1"].max(),
    )
    print(
        "z_rgb_features:",
        sample["z_rgb_features"].shape,
        sample["z_rgb_features"].dtype,
    )
    print("z_depth_indices:", sample["z_depth_indices"])
    print("depth1_path:", sample["depth1_path"])

    Path(results_folder).mkdir(parents=True, exist_ok=True)
    save_image(
        sample["depth1"][:1],
        f"{results_folder}/debug_depth1.png",
        normalize=True,
    )

    laq = LatentActionQuantizationStage25(
        dim=1024,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
    ).cuda()

    trainer = LAQStage25Trainer(
        laq,
        dataset=dataset,
        batch_size=args.batch_size,
        grad_accum_every=1,
        num_train_steps=args.num_train_steps,
        results_folder=results_folder,
        lr=args.lr,
        save_model_every=args.save_model_every,
        log_every=args.log_every,
    )



    trainer.train()


if __name__ == "__main__":
    main()